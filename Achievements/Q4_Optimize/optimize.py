import pandas as pd
import numpy as np
import itertools
import sys

# ==========================================
# 1. 配置与预处理
# ==========================================
DATA_FILE = '2026_MCM_Problem_C_Data_with_bayesian_weekly.csv'

# 损失函数权重偏好 (可以在这里调整价值观)
# Alpha: 能力损失权重 (重视技术)
# Beta:  参与度损失权重 (重视人气)
# Gamma: 无聊损失权重 (重视悬念/反转)
W_ALPHA = 1.0
W_BETA = 1.0
W_GAMMA = 1.0

def load_and_clean_data(filepath):
    """
    读取并清洗数据，处理空值和格式
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}。请确保文件在当前目录下。")
        sys.exit(1)

    # 将裁判分数和粉丝百分比转换为数值型
    for w in range(1, 13): # 假设最多12周
        # 处理粉丝数据
        fan_col = f'week{w}_Est_FanPct_Mean_Bayesian'
        if fan_col in df.columns:
            df[fan_col] = pd.to_numeric(df[fan_col], errors='coerce').fillna(0)
        
        # 处理裁判数据
        for j in range(1, 5):
            judge_col = f'week{w}_judge{j}_score'
            if judge_col in df.columns:
                df[judge_col] = pd.to_numeric(df[judge_col], errors='coerce').fillna(0)
    
    return df

# ==========================================
# 2. 核心算法：单周淘汰模拟
# ==========================================
def simulate_week_elimination(week_df, week_num, params):
    """
    模拟特定一周在给定赛制参数下的淘汰结果。
    
    参数:
    week_df: 当前周所有参赛选手的数据
    params: 字典, 包含 {'n_safe', 'lambda', 'w_judge', 'k_bottom'}
    
    返回:
    eliminated_row: 被淘汰选手的数据行 (Series)
    stats: 用于计算损失的统计信息
    """
    # ------------------------------------------
    # 步骤 1: 基础数据计算
    # ------------------------------------------
    # 计算裁判总分
    judge_cols = [c for c in week_df.columns if f'week{week_num}_judge' in c and 'score' in c]
    week_df = week_df.copy()
    week_df['total_judge_raw'] = week_df[judge_cols].sum(axis=1)
    
    # 过滤掉已经退赛或无分数的选手
    fan_col = f'week{week_num}_Est_FanPct_Mean_Bayesian'
    active_df = week_df[(week_df['total_judge_raw'] > 0) & (week_df[fan_col] > 0)].copy()
    
    n_contestants = len(active_df)
    if n_contestants <= params['k_bottom']: 
        return None, None # 人数太少，不足以进行常规淘汰逻辑

    # ------------------------------------------
    # 步骤 2: 应用“黄金豁免权” (Layer 1)
    # ------------------------------------------
    # 按裁判分降序排列
    active_df['judge_rank_dense'] = active_df['total_judge_raw'].rank(ascending=False, method='min')
    
    # 标记豁免选手
    # 注意：如果有并列第一，且 n_safe=1，通常规则是都豁免或都不豁免，这里简化为只要排名 <= n_safe 就豁免
    # 如果 n_safe=1，排名第1的都豁免
    safe_mask = active_df['judge_rank_dense'] <= params['n_safe']
    
    # 分离安全区和危险区选手
    safe_zone = active_df[safe_mask].copy()
    danger_zone = active_df[~safe_mask].copy()
    
    # 如果所有人都被豁免了(极端情况)，则取消豁免，所有人进入计分
    if danger_zone.empty:
        danger_zone = active_df.copy()

    # ------------------------------------------
    # 步骤 3: 混合计分 (Layer 2)
    # ------------------------------------------
    # 仅针对 Danger Zone 的选手计算排名和得分
    
    # A. 归一化裁判分 (0-100)
    max_j = danger_zone['total_judge_raw'].max()
    danger_zone['score_judge_norm'] = (danger_zone['total_judge_raw'] / max_j) * 100
    
    # B. 计算粉丝分 (混合 Lambda)
    # Rank部分: 粉丝排名越高(1), 分数越高。 
    # 转换逻辑: Score = (N - Rank + 1) / N * 100
    danger_zone['fan_rank_val'] = danger_zone[fan_col].rank(ascending=False, method='min')
    n_danger = len(danger_zone)
    danger_zone['score_fan_rank'] = (n_danger - danger_zone['fan_rank_val'] + 1) / n_danger * 100
    
    # Percent部分: 直接使用 (已经大致是 0-100 或 0-X)
    # 为了统一量纲，确保 fan_pct 也是 0-100 左右。原数据 fan_pct 是百分比(e.g. 11.5)
    danger_zone['score_fan_pct'] = danger_zone[fan_col] * (100 / danger_zone[fan_col].max()) # 归一化到最高者100
    
    # 混合
    lam = params['lambda']
    danger_zone['score_fan_mixed'] = (lam * danger_zone['score_fan_rank']) + ((1-lam) * danger_zone['score_fan_pct'])
    
    # C. 最终加权总分
    wj = params['w_judge']
    danger_zone['final_score'] = (wj * danger_zone['score_judge_norm']) + ((1-wj) * danger_zone['score_fan_mixed'])
    
    # ------------------------------------------
    # 步骤 4: 生死加赛 / 熔断 (Layer 3)
    # ------------------------------------------
    # 找出总分最低的 K 人 (Bottom K)
    # 按 final_score 升序
    danger_zone = danger_zone.sort_values(by='final_score', ascending=True)
    bottom_k_df = danger_zone.head(params['k_bottom'])
    
    # 裁判决定谁淘汰
    # 逻辑：在 Bottom K 中，裁判分数(total_judge_raw)最低的人被淘汰
    # 如果裁判分相同，则粉丝分低的淘汰
    eliminated = bottom_k_df.sort_values(by=['total_judge_raw', fan_col], ascending=[True, True]).iloc[0]
    
    # ------------------------------------------
    # 步骤 5: 收集统计数据 (用于 Loss Function)
    # ------------------------------------------
    # 我们需要知道这个被淘汰的人，在"全员"(含豁免者)中的排位水平
    
    # 全员裁判排名百分位 (0=第一名/最好, 1=最后一名/最差)
    # 使用 rank(pct=True)
    active_df['judge_pct_rank'] = active_df['total_judge_raw'].rank(ascending=False, pct=True)
    active_df['fan_pct_rank'] = active_df[fan_col].rank(ascending=False, pct=True)
    
    # 获取淘汰者的真实全场排名信息
    elim_stats = {
        'elim_name': eliminated['celebrity_name'],
        'judge_rank_pct': active_df.loc[eliminated.name, 'judge_pct_rank'], # 接近1表示淘汰了差的(好)，接近0表示淘汰了好的(坏)
        'fan_rank_pct': active_df.loc[eliminated.name, 'fan_pct_rank'],     # 接近1表示淘汰了没人气的(好)，接近0表示淘汰了顶流(坏)
        'is_judge_bottom': eliminated['total_judge_raw'] == active_df['total_judge_raw'].min() # 是否是全场裁判分最低
    }
    
    return eliminated, elim_stats

# ==========================================
# 3. 损失函数计算
# ==========================================
def calculate_system_loss(all_elim_stats):
    if not all_elim_stats:
        return 999.0, {}
    
    df = pd.DataFrame(all_elim_stats)
    
    # 1. Merit Loss (能力损失)
    # 目标：淘汰 judge_rank_pct 接近 1 的人。
    # 惩罚：如果淘汰了 judge_rank_pct 接近 0 的人 (高分选手)，Loss 应该很大。
    # 公式：(1 - rank_pct)^2
    # 例：淘汰了第1名(rank_pct=0.01) -> (1-0.01)^2 = 0.98 (极大损失)
    # 例：淘汰了最后名(rank_pct=1.00) -> (1-1.00)^2 = 0.00 (无损失)
    merit_loss = ((1 - df['judge_rank_pct']) ** 2).mean()
    
    # 2. Engagement Loss (流量损失)
    # 目标：淘汰 fan_rank_pct 接近 1 的人。
    # 惩罚：淘汰了顶流 (rank_pct 接近 0)
    engagement_loss = ((1 - df['fan_rank_pct']) ** 2).mean()
    
    # 3. Boredom Loss (无聊损失)
    # 目标：要有一定的逆转率 (Upset Rate)。
    # 如果每次淘汰的都是 is_judge_bottom (裁判眼中的倒数第一)，说明毫无悬念。
    # Upset Rate = (淘汰者不是裁判倒数第一的次数) / 总次数
    upset_rate = 1 - df['is_judge_bottom'].mean()
    
    # 设定理想的逆转率阈值，例如 0.25 (25%的时候应该发生一些意外)
    target_upset = 0.25
    # 如果 upset_rate < target，产生损失。使用 ReLU 逻辑
    boredom_loss = 0
    if upset_rate < target_upset:
        boredom_loss = (target_upset - upset_rate) ** 2
    else:
        # 如果逆转太多(>0.5)，可能意味着太混乱，也可以加一点惩罚，这里暂不加
        boredom_loss = 0
        
    # 4. 总损失 (加权)
    total_loss = (W_ALPHA * merit_loss) + \
                 (W_BETA * engagement_loss) + \
                 (W_GAMMA * boredom_loss)
                 
    return total_loss, {
        'Merit': merit_loss, 
        'Engagement': engagement_loss, 
        'Boredom': boredom_loss,
        'Upset_Rate': upset_rate
    }

# ==========================================
# 4. 优化器主程序 (Grid Search)
# ==========================================
def optimize_dmes():
    df = load_and_clean_data(DATA_FILE)
    seasons = sorted(df['season'].unique())
    
    # 定义超参数网格
    param_grid = {
        'n_safe': [0, 1],                # 0=无豁免, 1=Top1豁免
        'lambda': [0.1, 0.3, 0.5, 0.7, 0.9], # 0=纯百分比, 1=纯排名
        'w_judge': [0.1, 0.3, 0.5,0.7,0.9],                # 裁判权重通常固定0.5，也可优化 [0.4, 0.5, 0.6]
        'k_bottom': [2, 3]                  # 生死加赛人数固定为2
    }
    
    # 生成组合
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"正在分析 {len(combinations)} 种赛制组合，基于 {len(seasons)} 个历史赛季数据...")
    print("-" * 60)
    print(f"{'N_Safe':<6} | {'Lambda':<6} | {'W_Judge':<7} | {'Total Loss':<10} | {'Merit':<6} | {'Engage':<6} | {'Boredom':<7}")
    print("-" * 60)
    
    best_loss = float('inf')
    best_params = None
    best_details = None
    
    # 遍历每一种参数组合
    for params in combinations:
        all_season_stats = []
        
        # 遍历历史赛季 (回测)
        for season in seasons:
            season_df = df[df['season'] == season]
            
            # 遍历该赛季的每一周
            for week in range(1, 12): # 假设最多11周
                elim, stat = simulate_week_elimination(season_df, week, params)
                if stat:
                    all_season_stats.append(stat)
        
        # 计算该参数组合的总损失
        total_l, details = calculate_system_loss(all_season_stats)
        
        # 打印进度
        print(f"{params['n_safe']:<6} | {params['lambda']:<6.1f} | {params['w_judge']:<7.1f} | "
              f"{total_l:<10.4f} | {details['Merit']:<6.3f} | {details['Engagement']:<6.3f} | {details['Boredom']:<7.4f}")
        
        # 更新最优解
        if total_l < best_loss:
            best_loss = total_l
            best_params = params
            best_details = details

    print("-" * 60)
    print("\n>>> 最优赛制参数发现 (Optimal System Configuration) <<<")
    print(f"最小总损失 (Minimum Loss): {best_loss:.5f}")
    print("\n建议参数:")
    print(f"1. 黄金豁免 (N_Safe):    {best_params['n_safe']} (Top {best_params['n_safe']} safe from vote)")
    print(f"2. 混合系数 (Lambda):    {best_params['lambda']} ({best_params['lambda']*100:.0f}% Rank + {(1-best_params['lambda'])*100:.0f}% Percent)")
    print(f"3. 裁判权重 (W_Judge):   {best_params['w_judge']}")
    print(f"4. 熔断机制 (K_Bottom):  Bottom {best_params['k_bottom']} Dance-Off")
    
    print("\n性能指标:")
    print(f"- 能力损失 (Merit Loss):       {best_details['Merit']:.4f} (越低越好)")
    print(f"- 流量损失 (Engagement Loss):  {best_details['Engagement']:.4f} (越低越好)")
    print(f"- 逆转频率 (Upset Rate):       {best_details['Upset_Rate']*100:.1f}% (接近25%为佳)")

if __name__ == "__main__":
    optimize_dmes()