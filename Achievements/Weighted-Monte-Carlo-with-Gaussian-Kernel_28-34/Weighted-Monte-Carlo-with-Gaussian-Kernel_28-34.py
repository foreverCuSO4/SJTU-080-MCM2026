import pandas as pd
import numpy as np
import itertools
import math
from io import StringIO
import re
from tqdm import tqdm

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

# ---------------------------------------------------------
# 1. 全局配置参数
# ---------------------------------------------------------

# 贝叶斯权重参数：粉丝波动率（标准差），控制对时序一致性的敏感度
# 值越小表示越相信粉丝排名在相邻周之间保持稳定
SIGMA = 2.0

# 模拟迭代次数限制：
# - None: 严格全排列枚举（适用于选手人数少的情况）
# - 整数: 当 N! > ITERATIONS 时使用随机采样，否则使用全排列
# 例如：10000 表示最多采样1万次，如果全排列数少于1万则用全排列
ITERATIONS = 100000

# 不同赛季
target_seasons = [28,29,30,31,32,33,34]

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

def get_adjusted_expectations(prev_week_means, current_active_names):
    """
    修正幸存者偏差：当选手被淘汰后，幸存选手的排名会自然上移。
    本函数将上周的排名映射到当前周的期望排名（考虑淘汰选手的影响）。
    
    参数:
        prev_week_means: 上一周所有选手的粉丝排名估计字典 {name: mean_rank}
        current_active_names: 当前周仍在比赛的选手名字列表
    
    返回:
        adjusted_expectations: 调整后的期望排名字典 {name: expected_rank}
    
    示例:
        上周: {Bob: 2.1, Alice: 4.5, Charlie: 1.5}
        本周活跃: [Bob, Alice]  (Charlie被淘汰)
        输出: {Bob: 1.0, Alice: 2.0}  (Bob原本第2，Charlie淘汰后自然成为第1)
    """
    if prev_week_means is None or len(prev_week_means) == 0:
        # 如果没有上一周数据，返回空字典
        return {}
    
    # 1. 筛选出仍在比赛的选手及其上周排名
    surviving_ranks = {}
    for name in current_active_names:
        if name in prev_week_means:
            surviving_ranks[name] = prev_week_means[name]
    
    # 2. 如果没有幸存者（全是新选手，不太可能），返回空字典
    if len(surviving_ranks) == 0:
        return {}
    
    # 3. 按上周排名排序（升序，排名小的在前）
    sorted_survivors = sorted(surviving_ranks.items(), key=lambda x: x[1])
    
    # 4. 重新分配期望排名 1, 2, 3, ..., N
    adjusted_expectations = {}
    for new_rank, (name, _) in enumerate(sorted_survivors, start=1):
        adjusted_expectations[name] = float(new_rank)
    
    return adjusted_expectations

def run_simulation(season_num, week_num, df, prev_week_estimates=None, sigma=2.0, iterations=100000, has_judges_save=False):
    """
    基于贝叶斯加权/重要性采样的粉丝排名推断。
    
    参数:
        season_num: 赛季编号
        week_num: 周数
        df: 原始数据 DataFrame
        prev_week_estimates: 上一周的估计结果字典 {name: mean_rank}, 用于计算时序一致性权重
        sigma: 粉丝波动率（标准差），控制对时序一致性的敏感度
        iterations: 限制枚举的排列数（None=全排列）
        has_judges_save: 是否启用评委挽救规则（Season 28+ 为 True）
    
    返回:
        (result_df, current_week_estimates): 结果 DataFrame 和当周估计字典
    """
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
        return None, None

    # 2. 确定“谁是实际被淘汰者”
    # 解析 results 列，例如 "Eliminated Week 2"（必须精确匹配周数，避免 Week 1 匹配 Week 10/11）
    target_eliminated_name = None
    for _, row in active_df.iterrows():
        elim_week = extract_eliminated_week(row.get('results'))
        if elim_week == week_num:
            target_eliminated_name = row['celebrity_name']
            break
            
    # 如果这一周没有淘汰（比如第一周通常不淘汰），则跳过
    if target_eliminated_name is None:
        print(f"Season {season_num} Week {week_num}: 无淘汰记录，跳过模拟。")
        return None, None

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
        return None, None
    actual_loser_idx = int(actual_loser_idx[0])

    # 准备上一周估计（用于计算时序一致性权重）
    # 构建当前选手名到索引的映射
    name_to_idx = {active_df.loc[i, 'celebrity_name']: i for i in range(num_dancers)}
    
    # 修正幸存者偏差：计算调整后的期望排名
    # 当有选手被淘汰时，幸存选手的排名会自然上移，这不应该被视为"波动"
    current_active_names = [active_df.loc[i, 'celebrity_name'] for i in range(num_dancers)]
    adjusted_expectations = get_adjusted_expectations(prev_week_estimates, current_active_names)
    
    # 为避免存储所有有效排列（可能非常大），采用在线加权统计：
    # 对每个选手 i，累计 加权fan_rank 的和、加权平方和，以及 fan_rank==1 的加权次数
    valid_count = 0
    sum_weights = 0.0
    sum_w_ranks = np.zeros(num_dancers, dtype=float)      # Σ(w_i * rank_i)
    sum_w_ranksq = np.zeros(num_dancers, dtype=float)     # Σ(w_i * rank_i²)
    sum_w_first = np.zeros(num_dancers, dtype=float)      # Σ(w_i) if rank==1
    sum_wsq = 0.0  # Σ(w_i²) for ESS calculation
    total_checked = 0

    rank_values = np.arange(1, num_dancers + 1, dtype=int)
    total_perms = math.factorial(num_dancers)
    
    # 自适应策略：全排列数小于iterations则全排列，否则随机采样
    use_full_permutation = (iterations is None) or (total_perms <= iterations)
    
    if use_full_permutation:
        # 使用全排列
        progress_total = total_perms
        perm_iter = tqdm(
            itertools.permutations(rank_values),
            total=progress_total,
            desc=f"S{season_num}W{week_num} [全排列]",
            unit="perm"
        )
        
        for perm in perm_iter:
            total_checked += 1
            # perm 是 tuple，这里转成 numpy 方便向量化计算
            fan_ranks = np.fromiter(perm, dtype=int, count=num_dancers)

            # B. 计算综合排名
            # Rule: Total Rank = Judge Rank + Fan Rank
            total_ranks = judge_ranks + fan_ranks

            # C. 判定淘汰者 (综合排名数字最大的) - 硬约束
            if has_judges_save:
                # Season 28+：进入 Bottom 2 即可视为“可能被淘汰”
                # Bottom 2 阈值为倒数第二高的总排名，包含并列情况
                if len(total_ranks) >= 2:
                    threshold = np.sort(total_ranks)[-2]
                else:
                    threshold = np.max(total_ranks)
                simulated_losers_indices = np.where(total_ranks >= threshold)[0]
            else:
                max_val = np.max(total_ranks)
                simulated_losers_indices = np.where(total_ranks == max_val)[0]

            if actual_loser_idx in simulated_losers_indices:
                # D. 计算贝叶斯权重（基于与调整后期望排名的距离）
                weight = 1.0
                if len(adjusted_expectations) > 0:
                    # 计算当前排列与调整后期望的欧氏距离平方
                    distance_sq = 0.0
                    matched_count = 0
                    for name, expected_rank in adjusted_expectations.items():
                        if name in name_to_idx:
                            curr_idx = name_to_idx[name]
                            curr_rank = fan_ranks[curr_idx]
                            distance_sq += (curr_rank - expected_rank) ** 2
                            matched_count += 1
                    
                    if matched_count > 0:
                        weight = np.exp(-distance_sq / (2.0 * sigma ** 2))
                
                # E. 累加加权统计量
                valid_count += 1
                sum_weights += weight
                sum_wsq += weight ** 2
                sum_w_ranks += weight * fan_ranks
                sum_w_ranksq += weight * (fan_ranks ** 2)
                sum_w_first += weight * (fan_ranks == 1)
    else:
        # 使用随机采样
        progress_total = iterations
        perm_iter = tqdm(
            range(iterations),
            desc=f"S{season_num}W{week_num} [随机采样]",
            unit="perm"
        )
        
        for _ in perm_iter:
            total_checked += 1
            # 生成随机排列
            fan_ranks = np.random.permutation(rank_values)

            # B. 计算综合排名
            # Rule: Total Rank = Judge Rank + Fan Rank
            total_ranks = judge_ranks + fan_ranks

            # C. 判定淘汰者 (综合排名数字最大的) - 硬约束
            if has_judges_save:
                # Season 28+：进入 Bottom 2 即可视为“可能被淘汰”
                # Bottom 2 阈值为倒数第二高的总排名，包含并列情况
                if len(total_ranks) >= 2:
                    threshold = np.sort(total_ranks)[-2]
                else:
                    threshold = np.max(total_ranks)
                simulated_losers_indices = np.where(total_ranks >= threshold)[0]
            else:
                max_val = np.max(total_ranks)
                simulated_losers_indices = np.where(total_ranks == max_val)[0]

            if actual_loser_idx in simulated_losers_indices:
                # D. 计算贝叶斯权重（基于与调整后期望排名的距离）
                weight = 1.0
                if len(adjusted_expectations) > 0:
                    # 计算当前排列与调整后期望的欧氏距离平方
                    distance_sq = 0.0
                    matched_count = 0
                    for name, expected_rank in adjusted_expectations.items():
                        if name in name_to_idx:
                            curr_idx = name_to_idx[name]
                            curr_rank = fan_ranks[curr_idx]
                            distance_sq += (curr_rank - expected_rank) ** 2
                            matched_count += 1
                    
                    if matched_count > 0:
                        weight = np.exp(-distance_sq / (2.0 * sigma ** 2))
                
                # E. 累加加权统计量
                valid_count += 1
                sum_weights += weight
                sum_wsq += weight ** 2
                sum_w_ranks += weight * fan_ranks
                sum_w_ranksq += weight * (fan_ranks ** 2)
                sum_w_first += weight * (fan_ranks == 1)

    # 5. 统计结果
    if valid_count == 0 or sum_weights == 0.0:
        print("警告：没有找到符合历史结果的模拟情形（可能是模型约束太严或次数不够）。")
        return None, None

    # 计算加权均值、加权标准差、加权概率
    mean_ranks = sum_w_ranks / sum_weights  # 加权均值
    # 加权方差: Σ(w_i * x_i²)/Σw_i - μ²
    var_ranks = (sum_w_ranksq / sum_weights) - (mean_ranks ** 2)
    std_ranks = np.sqrt(np.maximum(var_ranks, 0.0))  # 加权标准差
    prob_first = sum_w_first / sum_weights  # 成为粉丝第1的加权概率
    
    # 计算有效样本量 (Effective Sample Size)
    ess = (sum_weights ** 2) / sum_wsq if sum_wsq > 0 else 0.0

    print(f"    Checked: {total_checked:,} | Valid: {valid_count:,} | Σweights: {sum_weights:.2f} | ESS: {ess:.1f}")

    results = []
    current_week_estimates = {}  # 保存当周估计，供下一周使用
    
    for i in range(num_dancers):
        name = active_df.loc[i, 'celebrity_name']
        avg_rank = mean_ranks[i]
        std_rank = std_ranks[i]
        
        # 保存到当周估计字典
        current_week_estimates[name] = avg_rank
        
        results.append({
            'Season': season_num,
            'Week': week_num,
            'Name': name,
            'Judge_Score': scores[i],
            'Judge_Rank': judge_ranks[i],
            'Est_Fan_Rank_Mean': round(avg_rank, 2),
            'Est_Fan_Rank_Std': round(std_rank, 2),
            'Prob_Fan_Favorite': round(prob_first[i] * 100, 1), # 百分比
            'Effective_Sample_Size': round(ess, 1),  # 新增ESS列
            'Actual_Result': 'Eliminated' if name == target_eliminated_name else 'Safe'
        })
        
    return pd.DataFrame(results), current_week_estimates


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

# 自动识别“有多少周就跑多少周”，并在每个赛季里淘汰到只剩 1 人时停止
weeks_to_simulate = get_available_weeks(df)

for s in target_seasons:
    prev_estimates = None  # 每个赛季开始时重置上一周估计
    for w in weeks_to_simulate:
        # 当周活跃选手 <= 1 视为赛季结束
        active_count = count_active_dancers(s, w, df)
        if active_count <= 1:
            print(f"Season {s} Week {w}: 活跃选手仅剩 {active_count} 人，停止该赛季模拟。")
            break
        has_judges_save = s >= 28
        res_df, curr_estimates = run_simulation(
            s, w, df,
            prev_week_estimates=prev_estimates,
            sigma=SIGMA,
            iterations=ITERATIONS,
            has_judges_save=has_judges_save
        )
        if res_df is not None:
            all_estimates.append(res_df)
            prev_estimates = curr_estimates  # 保存当周估计供下一周使用
        else:
            # 如果当周没有有效结果，重置估计
            prev_estimates = None

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