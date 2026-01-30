"""
百分比模式蒙特卡罗模拟：用于估计《与星共舞》第3-27赛季的粉丝投票百分比

在这些赛季中，淘汰规则基于评委百分比 + 粉丝投票百分比的组合。
本模块使用 Dirichlet 分布建模粉丝投票的连续概率分布。

数学模型：
- 评委百分比 (JP_i) = J_i / Σ(J) * 100
- 粉丝百分比 (FP_i) ~ Dirichlet(α) * 100，其中 Σ(FP_i) = 100
- 总分 (S_i) = JP_i + FP_i
- 约束：模拟有效当且仅当实际被淘汰者的总分最低

贝叶斯时序一致性（先验）：
- 第 t 周的先验来自第 t-1 周的后验
- 当选手被淘汰时，其粉丝按比例重新分配给幸存者
- α = μ * C，其中 C 是浓度参数（默认50.0）
- 第1周使用均匀先验：α = [1, 1, ..., 1]
"""

import pandas as pd
import numpy as np
import re
from tqdm import tqdm

# ---------------------------------------------------------
# 1. 全局配置参数
# ---------------------------------------------------------

# Dirichlet 浓度参数：控制粉丝投票的稳定性
# 值越大表示越相信粉丝投票在相邻周之间保持稳定
CONCENTRATION = 50.0

# 蒙特卡罗模拟迭代次数
ITERATIONS = 100000

# 赛季分组（根据不同的淘汰规则）
# Season 1-2: 排名制（严格淘汰最低分者）
SEASONS_RANK_STRICT = [1, 2]
# Season 3-27: 百分比制（评委% + 粉丝%）
SEASONS_PERCENTAGE = list(range(3, 28))  # [3, 4, 5, ..., 27]
# Season 28-34: 排名制 + 评委拯救（淘汰者在 Bottom 2 即可）
SEASONS_RANK_BOTTOM2 = list(range(28, 35))  # [28, 29, ..., 34]

# 目标赛季（所有34个赛季）
TARGET_SEASONS = SEASONS_RANK_STRICT + SEASONS_PERCENTAGE + SEASONS_RANK_BOTTOM2

# 用正则提取淘汰周数，避免 "Eliminated Week 1" 误匹配 "Eliminated Week 10/11"
_ELIM_WEEK_RE = re.compile(r"\bEliminated\s+Week\s+(\d+)\b", re.IGNORECASE)


def extract_eliminated_week(results_value):
    """从 results 字段中提取淘汰周数，提取不到返回 None。"""
    if results_value is None:
        return None
    s = str(results_value)
    m = _ELIM_WEEK_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def get_available_weeks(df):
    """从列名中自动识别数据里包含的周数（weekX_judgeY_score）。"""
    week_nums = set()
    for c in df.columns:
        m = re.match(r"week(\d+)_judge\d+_score", str(c))
        if m:
            week_nums.add(int(m.group(1)))
    return sorted(week_nums)


def count_active_dancers(season_num, week_num, df):
    """统计某赛季某周仍有分数（仍在比赛）的选手人数。"""
    cols = [f'week{week_num}_judge{j}_score' for j in range(1, 5)]
    week_df = df[df['season'] == season_num].copy()
    # 若列不存在（异常数据），视为 0 人
    for c in cols:
        if c not in week_df.columns:
            return 0
    week_df['total_judge_score'] = week_df[cols].sum(axis=1)
    return int((week_df['total_judge_score'] > 0).sum())


def calculate_rank_points(scores):
    """
    将分数转换为排名分（1 到 N），处理并列情况使用平均排名法。
    
    规则：
    - 最低分得 1 分，最高分得 N 分
    - 并列分数取平均排名（如两人并列最低，都得 (1+2)/2 = 1.5）
    
    参数:
        scores: numpy 数组，原始分数
    
    返回:
        rank_points: numpy 数组，排名分（1 到 N）
    """
    n = len(scores)
    # 获取排序后的索引
    sorted_indices = np.argsort(scores)
    
    # 初始化排名分
    rank_points = np.zeros(n)
    
    i = 0
    while i < n:
        # 找到所有与当前值相同的元素
        current_score = scores[sorted_indices[i]]
        j = i
        while j < n and scores[sorted_indices[j]] == current_score:
            j += 1
        
        # 计算平均排名（排名从 1 开始）
        # 位置 i 到 j-1 的排名是 (i+1) 到 j
        avg_rank = (i + 1 + j) / 2.0
        
        # 为所有并列的元素分配平均排名
        for k in range(i, j):
            rank_points[sorted_indices[k]] = avg_rank
        
        i = j
    
    return rank_points


def calculate_rank_points_vectorized(scores_matrix):
    """
    向量化版本：批量将分数矩阵转换为排名分。
    
    使用 argsort().argsort() 方法进行快速排名计算。
    注意：此方法不处理并列（并列时取较低排名），但对于连续随机数来说并列概率极低。
    
    参数:
        scores_matrix: numpy 数组，形状 (num_samples, num_dancers)
    
    返回:
        rank_points_matrix: numpy 数组，形状 (num_samples, num_dancers)
                           排名从 1（最低）到 N（最高）
    """
    # argsort().argsort() + 1 给出从 1 开始的排名
    # 第一次 argsort 给出排序后元素的原始索引
    # 第二次 argsort 给出每个元素在排序序列中的位置（即排名-1）
    return np.argsort(np.argsort(scores_matrix, axis=1), axis=1) + 1


def get_season_mode(season_num):
    """
    根据赛季编号确定使用的规则模式。
    
    返回:
        'rank_strict': Season 1-2，排名制严格淘汰
        'percentage': Season 3-27，百分比制
        'rank_bottom2': Season 28-34，排名制 + 评委拯救
    """
    if season_num in SEASONS_RANK_STRICT:
        return 'rank_strict'
    elif season_num in SEASONS_PERCENTAGE:
        return 'percentage'
    elif season_num in SEASONS_RANK_BOTTOM2:
        return 'rank_bottom2'
    else:
        # 默认使用百分比制
        return 'percentage'


def construct_dirichlet_prior(prev_week_estimates, current_active_names, concentration):
    """
    构造 Dirichlet 先验参数 α。
    
    贝叶斯时序一致性逻辑：
    1. 如果没有上一周估计（第1周），使用均匀先验 α = [1, 1, ..., 1]
    2. 否则，从上一周估计中提取幸存者的均值，重新归一化后乘以浓度参数
    
    重归一化逻辑（处理淘汰选手的粉丝重新分配）：
    - 假设被淘汰选手的粉丝按比例重新分配给幸存者
    - 取上一周幸存者的均值，归一化使其和为 1.0
    - α = μ_normalized * concentration
    
    参数:
        prev_week_estimates: 上一周估计字典 {name: mean_fan_percent}，可以为 None
        current_active_names: 当前周活跃选手名字列表
        concentration: Dirichlet 浓度参数 C
    
    返回:
        alpha: numpy 数组，与 current_active_names 顺序对应的 α 参数
    """
    num_dancers = len(current_active_names)
    
    # Case 1: 没有上一周数据（第1周）或为空，使用均匀先验
    if prev_week_estimates is None or len(prev_week_estimates) == 0:
        # 均匀 Dirichlet 先验：α = [1, 1, ..., 1]
        return np.ones(num_dancers)
    
    # Case 2: 有上一周数据，构造信息性先验
    # 提取当前幸存者在上一周的估计值
    mu = np.zeros(num_dancers)
    has_prior = np.zeros(num_dancers, dtype=bool)
    
    for i, name in enumerate(current_active_names):
        if name in prev_week_estimates:
            mu[i] = prev_week_estimates[name]
            has_prior[i] = True
    
    # 处理新选手（上一周没有数据的情况，虽然在 DWTS 中很少见）
    # 为新选手分配幸存者平均值作为默认先验
    if not np.all(has_prior):
        existing_mean = np.mean(mu[has_prior]) if np.any(has_prior) else 100.0 / num_dancers
        mu[~has_prior] = existing_mean
    
    # 重归一化：确保 μ 和为 1.0
    # 这一步模拟了被淘汰选手粉丝按比例重新分配的过程
    total = np.sum(mu)
    if total > 0:
        mu_normalized = mu / total
    else:
        # 异常情况：所有值为0，使用均匀分布
        mu_normalized = np.ones(num_dancers) / num_dancers
    
    # 计算 Dirichlet 参数：α = μ * C
    # 较大的 C 值意味着分布更集中在 μ 附近（粉丝更稳定）
    alpha = mu_normalized * concentration
    
    # 确保 α > 0（Dirichlet 分布的要求）
    # 使用小的 epsilon 避免数值问题
    alpha = np.maximum(alpha, 1e-6)
    
    return alpha


def run_simulation_percent_mode(season_num, week_num, df, prev_week_estimates=None, 
                                 concentration=50.0, iterations=100000):
    """
    基于 Dirichlet 分布的粉丝投票百分比推断（支持所有 34 个赛季）。
    
    根据赛季自动选择规则模式：
    - Season 1-2: 排名制（严格淘汰最低分者）
    - Season 3-27: 百分比制（评委% + 粉丝%）
    - Season 28-34: 排名制 + 评委拯救（淘汰者在 Bottom 2 即可）
    
    数学模型：
    【百分比制 Season 3-27】
    - 评委百分比 JP_i = (J_i / Σ J) * 100
    - 粉丝百分比 FP ~ Dirichlet(α) * 100
    - 总分 S_i = JP_i + FP_i
    - 有效约束：实际淘汰者的 S 必须是最小的
    
    【排名制 Season 1-2, 28-34】
    - 评委排名分 JR_i = Rank(J_i)，1（最低）到 N（最高），并列取平均
    - 粉丝排名分 FR_i = Rank(FP_i)，1（最低）到 N（最高）
    - 总分 S_i = JR_i + FR_i
    - 有效约束：
      - Season 1-2: 淘汰者 S 必须是最小的
      - Season 28-34: 淘汰者 S 必须在 Bottom 2
    
    参数:
        season_num: 赛季编号
        week_num: 周数
        df: 原始数据 DataFrame
        prev_week_estimates: 上一周估计字典 {name: mean_fan_percent}
        concentration: Dirichlet 浓度参数 C（默认 50.0）
        iterations: 蒙特卡罗模拟次数
    
    返回:
        (result_df, current_week_estimates): 结果 DataFrame 和当周估计字典
    """
    # ---------------------------------------------------------
    # Step 0: 确定规则模式
    # ---------------------------------------------------------
    mode = get_season_mode(season_num)
    
    # ---------------------------------------------------------
    # Step 1: 数据准备 - 提取活跃选手和评委分数
    # ---------------------------------------------------------
    cols = [f'week{week_num}_judge{j}_score' for j in range(1, 5)]
    
    # 提取该赛季数据
    season_df = df[df['season'] == season_num].copy()
    
    # 计算当周评委总分
    season_df['total_judge_score'] = season_df[cols].sum(axis=1)
    
    # 筛选仍在比赛的选手（分数 > 0）
    active_df = season_df[season_df['total_judge_score'] > 0].reset_index(drop=True)
    
    if len(active_df) == 0:
        return None, None
    
    num_dancers = len(active_df)
    
    # 提取选手名字和评委原始分数
    names = active_df['celebrity_name'].tolist()
    judge_raw_scores = active_df['total_judge_score'].values.astype(float)
    
    # ---------------------------------------------------------
    # Step 2: 根据模式计算评委得分
    # ---------------------------------------------------------
    if mode == 'percentage':
        # 百分比制：JP_i = (J_i / Σ J) * 100
        total_judge_score = np.sum(judge_raw_scores)
        judge_scores_for_calc = (judge_raw_scores / total_judge_score) * 100.0
        judge_display_type = 'Judge_Percent'
    else:
        # 排名制：使用排名分（处理并列）
        judge_scores_for_calc = calculate_rank_points(judge_raw_scores)
        judge_display_type = 'Judge_Rank_Points'
    
    # ---------------------------------------------------------
    # Step 3: 确定实际淘汰者
    # ---------------------------------------------------------
    actual_eliminated_name = None
    for _, row in active_df.iterrows():
        elim_week = extract_eliminated_week(row.get('results'))
        if elim_week == week_num:
            actual_eliminated_name = row['celebrity_name']
            break
    
    # 找到淘汰者索引（如果有的话）
    actual_loser_idx = None
    if actual_eliminated_name is not None:
        for i, name in enumerate(names):
            if name == actual_eliminated_name:
                actual_loser_idx = i
                break
    
    # 判断本周是否有淘汰
    has_elimination = actual_loser_idx is not None
    
    mode_display = {
        'rank_strict': '排名制(严格)',
        'percentage': '百分比制',
        'rank_bottom2': '排名制(Bottom2)'
    }
    
    if not has_elimination:
        print(f"Season {season_num} Week {week_num} [{mode_display[mode]}]: 无淘汰记录，所有样本均有效。")
    else:
        print(f"--- Processing Season {season_num} Week {week_num} [{mode_display[mode]}] (Target: {actual_eliminated_name}) ---")
    
    # ---------------------------------------------------------
    # Step 4: 构造 Dirichlet 先验参数 α
    # ---------------------------------------------------------
    alpha = construct_dirichlet_prior(prev_week_estimates, names, concentration)
    
    # ---------------------------------------------------------
    # Step 5: 蒙特卡罗模拟（向量化）
    # ---------------------------------------------------------
    # 生成 iterations 个 Dirichlet 样本，每个样本是一个长度为 num_dancers 的向量
    # 形状: (iterations, num_dancers)
    fan_percent_samples = np.random.dirichlet(alpha, size=iterations) * 100.0
    
    # ---------------------------------------------------------
    # Step 6: 根据模式计算总分并进行拒绝采样
    # ---------------------------------------------------------
    if mode == 'percentage':
        # 百分比制：总分 = 评委百分比 + 粉丝百分比
        # JP 是固定的向量，需要广播
        total_scores = judge_scores_for_calc + fan_percent_samples
        
        if has_elimination:
            # 有效样本：实际淘汰者是总分最低的
            min_score_indices = np.argmin(total_scores, axis=1)
            valid_mask = (min_score_indices == actual_loser_idx)
        else:
            valid_mask = np.ones(iterations, dtype=bool)
    
    else:
        # 排名制（rank_strict 或 rank_bottom2）
        # 计算粉丝排名分（向量化）
        fan_rank_points = calculate_rank_points_vectorized(fan_percent_samples)
        
        # 总分 = 评委排名分 + 粉丝排名分
        total_scores = judge_scores_for_calc + fan_rank_points
        
        if has_elimination:
            if mode == 'rank_strict':
                # Season 1-2: 淘汰者必须是总分最低的
                min_score_indices = np.argmin(total_scores, axis=1)
                valid_mask = (min_score_indices == actual_loser_idx)
            else:
                # Season 28-34 (rank_bottom2): 淘汰者必须在 Bottom 2
                # 找到每个样本中总分最低的两个选手
                # partition 比完全排序更快
                if num_dancers >= 2:
                    # 获取最小的两个值的索引
                    # argpartition 返回使得第 k 个位置之前都是最小的 k 个元素的索引
                    partitioned_indices = np.argpartition(total_scores, 1, axis=1)
                    bottom2_indices = partitioned_indices[:, :2]  # 取前两列（最小的两个）
                    
                    # 检查淘汰者是否在 bottom 2 中
                    valid_mask = np.any(bottom2_indices == actual_loser_idx, axis=1)
                else:
                    # 只有 1 个选手，无法淘汰
                    valid_mask = np.ones(iterations, dtype=bool)
        else:
            valid_mask = np.ones(iterations, dtype=bool)
    
    # 提取有效样本
    valid_fan_percents = fan_percent_samples[valid_mask]
    valid_count = np.sum(valid_mask)
    
    # ---------------------------------------------------------
    # Step 7: 统计分析
    # ---------------------------------------------------------
    if valid_count == 0:
        print(f"    警告：没有找到符合历史结果的有效样本（concentration={concentration}）。")
        print(f"    尝试放宽约束或增加迭代次数。")
        return None, None
    
    acceptance_rate = valid_count / iterations * 100.0
    print(f"    Iterations: {iterations:,} | Valid: {valid_count:,} | Acceptance Rate: {acceptance_rate:.2f}%")
    
    # 计算经验均值和标准差
    mean_fan_percents = np.mean(valid_fan_percents, axis=0)
    std_fan_percents = np.std(valid_fan_percents, axis=0)
    
    # ---------------------------------------------------------
    # Step 8: 构建输出
    # ---------------------------------------------------------
    results = []
    current_week_estimates = {}
    
    for i in range(num_dancers):
        name = names[i]
        
        # 保存当周估计供下一周使用
        current_week_estimates[name] = mean_fan_percents[i]
        
        result_row = {
            'Season': season_num,
            'Week': week_num,
            'Name': name,
            'Judge_Raw_Score': judge_raw_scores[i],
            'Mode': mode,
            'Est_Fan_Percent_Mean': round(mean_fan_percents[i], 2),
            'Est_Fan_Percent_Std': round(std_fan_percents[i], 2),
            'Acceptance_Rate': round(acceptance_rate, 2),
            'Actual_Result': 'Eliminated' if name == actual_eliminated_name else 'Safe'
        }
        
        # 根据模式添加不同的评分列
        if mode == 'percentage':
            result_row['Judge_Percent'] = round(judge_scores_for_calc[i], 2)
        else:
            result_row['Judge_Rank_Points'] = round(judge_scores_for_calc[i], 2)
        
        results.append(result_row)
    
    return pd.DataFrame(results), current_week_estimates


# ---------------------------------------------------------
# 主程序
# ---------------------------------------------------------

if __name__ == "__main__":
    # 加载数据
    df = pd.read_csv('2026_MCM_Problem_C_Data.csv')
    
    # 数据清洗：将 N/A 转为 0，并将分数转为浮点数
    score_cols = [c for c in df.columns if 'score' in c]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 存储所有结果
    all_estimates = []
    
    # 自动识别可用周数
    weeks_to_simulate = get_available_weeks(df)
    
    print("=" * 70)
    print("蒙特卡罗模拟：Season 1-34 粉丝投票百分比估算")
    print(f"配置：Concentration={CONCENTRATION}, Iterations={ITERATIONS}")
    print("规则模式：")
    print(f"  - Season 1-2: 排名制（严格淘汰最低分者）")
    print(f"  - Season 3-27: 百分比制（评委% + 粉丝%）")
    print(f"  - Season 28-34: 排名制 + 评委拯救（Bottom 2）")
    print("=" * 70)
    
    # 遍历目标赛季
    for season in TARGET_SEASONS:
        print(f"\n{'='*50}")
        print(f"Season {season}")
        print(f"{'='*50}")
        
        prev_estimates = None  # 每个赛季开始时重置
        
        for week in weeks_to_simulate:
            # 检查当周是否还有足够的选手
            active_count = count_active_dancers(season, week, df)
            if active_count <= 1:
                print(f"Season {season} Week {week}: 活跃选手仅剩 {active_count} 人，停止该赛季模拟。")
                break
            
            # 运行模拟
            result_df, curr_estimates = run_simulation_percent_mode(
                season_num=season,
                week_num=week,
                df=df,
                prev_week_estimates=prev_estimates,
                concentration=CONCENTRATION,
                iterations=ITERATIONS
            )
            
            if result_df is not None:
                all_estimates.append(result_df)
                prev_estimates = curr_estimates
            else:
                # 如果当周没有有效结果，保持上一周的估计（或重置）
                # 这里选择保持，以便连续性
                pass
    
    # ---------------------------------------------------------
    # 汇总并输出结果
    # ---------------------------------------------------------
    if all_estimates:
        final_report = pd.concat(all_estimates, ignore_index=True)
        
        print("\n" + "=" * 70)
        print("蒙特卡罗模拟结果：粉丝投票百分比估算")
        print("=" * 70)
        
        # 调整显示设置
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', 50)
        
        # 显示部分结果（包含不同模式的列）
        display_cols = ['Season', 'Week', 'Name', 'Mode', 
                       'Est_Fan_Percent_Mean', 'Est_Fan_Percent_Std', 'Actual_Result']
        print(final_report[display_cols].head(30))
        
        # 导出结果
        output_csv = "monte_carlo_results_percent_mode.csv"
        final_report.to_csv(output_csv, index=False)
        print(f"\n结果已保存到: {output_csv}")
        
        # 打印一些统计摘要
        print("\n" + "=" * 70)
        print("统计摘要")
        print("=" * 70)
        
        # 按赛季统计平均接受率
        summary = final_report.groupby(['Season', 'Mode']).agg({
            'Acceptance_Rate': 'mean',
            'Week': 'max'
        }).round(2)
        summary.columns = ['Avg_Acceptance_Rate', 'Max_Week']
        print("\n每赛季统计：")
        print(summary)
        
        # 按模式统计
        mode_summary = final_report.groupby('Mode').agg({
            'Acceptance_Rate': 'mean',
            'Season': 'nunique'
        }).round(2)
        mode_summary.columns = ['Avg_Acceptance_Rate', 'Num_Seasons']
        print("\n按模式统计：")
        print(mode_summary)
        
        # 检查被淘汰者的预测准确性
        eliminated = final_report[final_report['Actual_Result'] == 'Eliminated'].copy()
        if len(eliminated) > 0:
            # 对于每次淘汰，检查被淘汰者是否有最低的预测粉丝支持
            print(f"\n淘汰事件总数: {len(eliminated)}")
            print(f"平均淘汰者粉丝百分比估计: {eliminated['Est_Fan_Percent_Mean'].mean():.2f}%")
    else:
        print("没有生成有效数据。")
