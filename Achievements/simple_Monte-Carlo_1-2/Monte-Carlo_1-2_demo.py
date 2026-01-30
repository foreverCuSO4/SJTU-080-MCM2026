import pandas as pd
import numpy as np
import itertools
import math
from io import StringIO
import re
from tqdm import tqdm



df = pd.read_csv('2026_MCM_Problem_C_Data.csv')


# 数据清洗：将N/A转为0或NaN，并将分数转为浮点数
score_cols = [c for c in df.columns if 'score' in c]
for col in score_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ---------------------------------------------------------
# 2. 定义辅助函数
# ---------------------------------------------------------

def get_judge_ranks(scores):
    """
    输入: 分数列表 (例如 [25, 30, 20])
    输出: 排名 (例如 [2, 1, 3])
    规则: 分数高 = Rank 1 (Dense Ranking, 这里的实现用scipy的rankdata类似逻辑)
    """
    # 将分数取负，因为argsort默认升序，而我们需要分数高的排前面
    # argsort两次可以得到排名 (0-indexed)
    ranks = np.argsort(np.argsort(-np.array(scores))) + 1
    # 注意：DWTS处理并列通常是占位，这里简化处理，直接用全排列名次
    # 如果需要处理并列（如同分共享Rank 2），逻辑会更复杂，此处为标准模型
    return ranks

def run_simulation(season_num, week_num, df, iterations=None):
    # 1. 筛选本赛季、本周的活跃选手
    # 逻辑：查找该周分数 > 0 的选手
    cols = [f'week{week_num}_judge{j}_score' for j in range(1, 5)]
    
    # 提取该周数据
    week_df = df[df['season'] == season_num].copy()
    
    # 计算当周评委总分
    week_df['total_judge_score'] = week_df[cols].sum(axis=1)
    
    # 筛选仍在比赛的选手（分数为0表示已淘汰或未开始）
    active_df = week_df[week_df['total_judge_score'] > 0].reset_index(drop=True)
    
    if len(active_df) == 0:
        return None

    # 2. 确定“谁是实际被淘汰者”
    # 解析 results 列，例如 "Eliminated Week 2"
    target_eliminated_name = None
    for idx, row in active_df.iterrows():
        res = str(row['results'])
        # 匹配 "Eliminated Week X" 且 X 等于当前周
        if f"Eliminated Week {week_num}" in res:
            target_eliminated_name = row['celebrity_name']
            break
            
    # 如果这一周没有淘汰（比如第一周通常不淘汰），则跳过
    if target_eliminated_name is None:
        print(f"Season {season_num} Week {week_num}: 无淘汰记录，跳过模拟。")
        return None

    print(f"--- Processing Season {season_num} Week {week_num} (Target Elimination: {target_eliminated_name}) ---")

    # 3. 计算评委排名 (Rank J)
    # 处理并列：使用'min' method，即如果有两人第一，则都是Rank 1，下一个人Rank 3
    # 但为了简化全排列模拟，我们暂时使用 'dense' 排名或直接基于分数的排序索引
    # 此处使用简单的 argsort 生成 1..N 的唯一排名（假设并列很少或不影响大局，严谨版需处理并列）
    # 更严谨的做法：分数相同，Rank J 相同
    scores = active_df['total_judge_score'].values
    # 简单的倒序排名 (分数高的排第1)
    # 使用 scipy.stats.rankdata 会更好，但在纯numpy下手动实现:
    temp = np.argsort(-scores) # 索引排序
    judge_ranks = np.empty_like(temp)
    judge_ranks[temp] = np.arange(1, len(scores) + 1)
    
    # 4. 枚举全排列 (Rejection Sampling)
    num_dancers = len(active_df)
    # 注意：当 num_dancers 较大时，全排列数量为 N!，可能非常耗时。
    # 默认 iterations=None 表示严格枚举所有排列；若传入整数，则仅枚举前 iterations 个排列（调试/限时用）。

    # 找到实际淘汰者在 active_df 中的索引
    actual_loser_idx = active_df[active_df['celebrity_name'] == target_eliminated_name].index
    if len(actual_loser_idx) == 0:
        print("警告：未在当周活跃选手中找到实际淘汰者，跳过。")
        return None
    actual_loser_idx = int(actual_loser_idx[0])

    # 为避免存储所有有效排列（可能非常大），采用在线统计：
    # 对每个选手 i，累计 fan_rank 的和/平方和，以及 fan_rank==1 的次数
    valid_count = 0
    sum_ranks = np.zeros(num_dancers, dtype=float)
    sumsq_ranks = np.zeros(num_dancers, dtype=float)
    count_first = np.zeros(num_dancers, dtype=int)
    total_checked = 0

    rank_values = np.arange(1, num_dancers + 1, dtype=int)
    total_perms = math.factorial(num_dancers)
    progress_total = total_perms if iterations is None else min(iterations, total_perms)
    perm_iter = tqdm(
        itertools.permutations(rank_values),
        total=progress_total,
        desc=f"Season {season_num} Week {week_num}",
        unit="perm"
    )

    for perm in perm_iter:
        if iterations is not None and total_checked >= iterations:
            break
        total_checked += 1

        # perm 是 tuple，这里转成 numpy 方便向量化计算
        fan_ranks = np.fromiter(perm, dtype=int, count=num_dancers)

        # B. 计算综合排名
        # Rule: Total Rank = Judge Rank + Fan Rank
        total_ranks = judge_ranks + fan_ranks

        # C. 判定淘汰者 (综合排名数字最大的)
        # 注意：如果有并列最后一名，现实中需要Tie-Breaker (通常看评委分)。
        # 这里简化：只要实际淘汰者的总排名是 Max 之一，就算有效。
        max_val = np.max(total_ranks)
        simulated_losers_indices = np.where(total_ranks == max_val)[0]

        if actual_loser_idx in simulated_losers_indices:
            valid_count += 1
            sum_ranks += fan_ranks
            sumsq_ranks += fan_ranks * fan_ranks
            count_first += (fan_ranks == 1)

    # 5. 统计结果
    if valid_count == 0:
        print("警告：没有找到符合历史结果的模拟情形（可能是模型约束太严或次数不够）。")
        return None

    # 计算每个选手粉丝排名的均值/标准差，以及成为粉丝第1的概率
    mean_ranks = sum_ranks / valid_count
    var_ranks = (sumsq_ranks / valid_count) - (mean_ranks ** 2)
    std_ranks = np.sqrt(np.maximum(var_ranks, 0.0))
    prob_first = count_first / valid_count

    print(f"    Checked permutations: {total_checked:,} | Valid (match elimination): {valid_count:,}")

    results = []
    for i in range(num_dancers):
        name = active_df.loc[i, 'celebrity_name']
        avg_rank = mean_ranks[i]
        std_rank = std_ranks[i]
        
        results.append({
            'Season': season_num,
            'Week': week_num,
            'Name': name,
            'Judge_Score': scores[i],
            'Judge_Rank': judge_ranks[i],
            'Est_Fan_Rank_Mean': round(avg_rank, 2),
            'Est_Fan_Rank_Std': round(std_rank, 2),
            'Prob_Fan_Favorite': round(prob_first[i] * 100, 1), # 百分比
            'Actual_Result': 'Eliminated' if name == target_eliminated_name else 'Safe'
        })
        
    return pd.DataFrame(results)


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

# ---------------------------------------------------------
# 3. 运行程序
# ---------------------------------------------------------

all_estimates = []

# 针对 Season 1 和 2 (简单排名制)
target_seasons = [1, 2]

# 自动识别“有多少周就跑多少周”，并在每个赛季里淘汰到只剩 1 人时停止
weeks_to_simulate = get_available_weeks(df)

for s in target_seasons:
    for w in weeks_to_simulate:
        # 当周活跃选手 <= 1 视为赛季结束
        active_count = count_active_dancers(s, w, df)
        if active_count <= 1:
            print(f"Season {s} Week {w}: 活跃选手仅剩 {active_count} 人，停止该赛季模拟。")
            break
        res_df = run_simulation(s, w, df)
        if res_df is not None:
            all_estimates.append(res_df)

# 合并所有结果
if all_estimates:
    final_report = pd.concat(all_estimates)
    
    # 打印结果表
    print("\n====== 蒙特卡罗模拟结果：粉丝排名估算 ======")
    # 调整显示列宽
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(final_report[['Season', 'Week', 'Name', 'Judge_Rank', 'Est_Fan_Rank_Mean', 'Est_Fan_Rank_Std', 'Actual_Result']])
    
    # 示例分析：查看某次淘汰中，是否有争议
    print("\n[分析示例]：Season 1 Week 4 (Rachel Hunter 被淘汰)")
    print(final_report[(final_report['Season']==1) & (final_report['Week']==4)])

    # 导出结果到 CSV
    output_csv = "monte_carlo_results.csv"
    final_report.to_csv(output_csv, index=False)
    print(f"\n结果已保存到: {output_csv}")
else:
    print("没有生成有效数据。")