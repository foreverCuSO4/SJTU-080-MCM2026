"""simulation_percent_mode_Grid_search.py

专用网格搜索脚本：搜索 Dirichlet 先验混合权重 inertia_weight。

与 Achievements/model_mixed_with_google_trends_stable/simulation_percent_mode.py
保持一致的“命中率/命中”统计口径：

- 将“命中”定义为：某周存在至少一个有效样本（valid_count > 0）。
- 将“没命中 (no-hit)”定义为：valid_count == 0，即 run_simulation_percent_mode 返回 result_df=None。
- 同时统计：
    * 所有周：weeks_attempted / weeks_no_hit
    * 仅淘汰周：elim_weeks_attempted / elim_weeks_no_hit

注意：为了与 stable 版本完全一致，本脚本默认不固定随机种子。
"""

import pandas as pd
import numpy as np
import re
import argparse
from scipy.stats import rankdata
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# ---------------------------------------------------------
# 1. 全局配置参数
# ---------------------------------------------------------

# 热度系数文件路径
POPULARITY_FILE = 'popularity_coefficients.csv'

# Dirichlet 浓度参数：控制粉丝投票的稳定性
# 值越大表示越相信粉丝投票在相邻周之间保持稳定
CONCENTRATION = 50.0

# 贝叶斯先验混合权重（更“合理”的模型）：
# - 惯性分量：来自上一周后验（或第1周的热度先验）
# - 路人观众分量：始终是均匀分布 (1/N)
# inertia_weight 越大，越相信“惯性”；越小，越相信“路人随机性”。
# 默认惯性权重（网格搜索会覆盖该值；保留仅用于“非网格搜索”场景）
INERTIA_WEIGHT = 0.6

# 贝叶斯先验更新策略：
# - True : 贝叶斯先验不再随周更新；每周都使用“第1周先验”（Google Trends 热度 + 均匀先验按 INERTIA_WEIGHT 加权）
# - False: 使用上一周后验作为下一周先验（时序一致性更新）
BAYESIAN_STATIC_WEEK1_PRIOR = False

# CSV 导出时的分类值映射
CSV_VALUE_MAP_MODE = {
    'rank_strict': 'Rank (Strict Elimination)',
    'percentage': 'Percentage (Judge% + Fan%)',
    'rank_bottom2': 'Rank (Bottom 2 Elimination)',
}

# 规则模式显示名称
MODE_DISPLAY = {
    'rank_strict': '排名制(严格)',
    'percentage': '百分比制',
    'rank_bottom2': '排名制(Bottom2)'
}

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
    """将分数转换为排名分（1到N），并列取平均排名。"""
    return rankdata(scores, method='average')


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
    """根据赛季编号确定使用的规则模式。"""
    if season_num <= 2:
        return 'rank_strict'
    elif season_num <= 27:
        return 'percentage'
    elif season_num <= 34:
        return 'rank_bottom2'
    return 'percentage'


def construct_dirichlet_prior(prev_week_estimates, current_active_names, concentration,
                              use_bayesian=True, popularity_data=None, season_num=None,
                              inertia_weight=0.85, verbose=True,
                              force_first_week_prior=False):
    """
    构造 Dirichlet 先验参数 α。
    
    贝叶斯时序一致性逻辑：
    1. 如果没有上一周估计（第1周），使用 Google Trends 热度数据作为先验
       - 若热度数据可用，α = popularity_normalized * concentration
       - 若热度数据不可用，退化为均匀先验 α = [1, 1, ..., 1]
    2. 否则，从上一周估计中提取幸存者的均值，重新归一化后乘以浓度参数
    
    重归一化逻辑（处理淘汰选手的粉丝重新分配）：
    - 假设被淘汰选手的粉丝按比例重新分配给幸存者
    - 取上一周幸存者的均值，归一化使其和为 1.0
    - α = μ_normalized * concentration
    
    参数:
        prev_week_estimates: 上一周估计字典 {name: mean_fan_percent}，可以为 None
        current_active_names: 当前周活跃选手名字列表
        concentration: Dirichlet 浓度参数 C
        use_bayesian: 是否使用贝叶斯时序一致性
        popularity_data: 热度数据 DataFrame，包含 season, celebrity_name, popularity_coefficient
        season_num: 当前赛季编号
    
    返回:
        alpha: numpy 数组，与 current_active_names 顺序对应的 α 参数
    """
    num_dancers = len(current_active_names)

    # 可选：关闭贝叶斯时序一致性时，每周固定使用均匀先验
    if not use_bayesian:
        return np.ones(num_dancers)
    
    # 先构造“惯性分量”的均值向量 mu_inertia（随后再与均匀分量混合）
    mu_inertia = None
    has_source_coverage = None  # 仅用于打印覆盖率

    # Case 1: 第1周（或强制使用第1周先验）：惯性分量来自热度先验（若可用）
    if force_first_week_prior or prev_week_estimates is None or len(prev_week_estimates) == 0:
        if popularity_data is not None and season_num is not None:
            season_pop = popularity_data[popularity_data['season'] == season_num]
            if len(season_pop) > 0:
                mu = np.zeros(num_dancers)
                has_popularity = np.zeros(num_dancers, dtype=bool)

                for i, name in enumerate(current_active_names):
                    match = season_pop[season_pop['celebrity_name'] == name]
                    if len(match) > 0:
                        mu[i] = match['popularity_coefficient'].values[0]
                        has_popularity[i] = True

                if np.any(has_popularity):
                    existing_mean = np.mean(mu[has_popularity])
                    mu[~has_popularity] = existing_mean if existing_mean > 0 else 1.0 / num_dancers
                else:
                    mu = np.ones(num_dancers)

                mu = np.maximum(mu, 1e-6)
                mu_inertia = mu
                has_source_coverage = has_popularity
        
        # 若热度不可用，则惯性分量退化为均匀（与路人分量一致）
        if mu_inertia is None:
            mu_inertia = np.ones(num_dancers)
            has_source_coverage = np.zeros(num_dancers, dtype=bool)

    # Case 2: 有上一周数据：惯性分量来自上一周后验均值（并会自动对幸存者重归一化）
    else:
        mu = np.zeros(num_dancers)
        has_prior = np.zeros(num_dancers, dtype=bool)
        for i, name in enumerate(current_active_names):
            if name in prev_week_estimates:
                mu[i] = prev_week_estimates[name]
                has_prior[i] = True

        if not np.all(has_prior):
            existing_mean = np.mean(mu[has_prior]) if np.any(has_prior) else 100.0 / num_dancers
            mu[~has_prior] = existing_mean

        mu = np.maximum(mu, 1e-6)
        mu_inertia = mu
        has_source_coverage = has_prior

    # --------------------------
    # 混合：mu = w * mu_inertia_norm + (1-w) * uniform
    # --------------------------
    w = float(inertia_weight)
    w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)

    total_inertia = np.sum(mu_inertia)
    if total_inertia > 0:
        mu_inertia_norm = mu_inertia / total_inertia
    else:
        mu_inertia_norm = np.ones(num_dancers) / num_dancers

    mu_uniform = np.ones(num_dancers) / num_dancers
    mu_mixed = w * mu_inertia_norm + (1.0 - w) * mu_uniform

    # 数值防护并再次归一化（确保和为1）
    mu_mixed = np.maximum(mu_mixed, 1e-12)
    mu_mixed = mu_mixed / np.sum(mu_mixed)

    alpha = mu_mixed * concentration
    alpha = np.maximum(alpha, 1e-6)

    if force_first_week_prior:
        # 强制每周使用“第1周先验”（热度，可能退化为均匀）
        covered = int(np.sum(has_source_coverage)) if has_source_coverage is not None else 0
        if verbose:
            print(f"    [先验] 固定先验: 惯性(热度) + 路人(均匀) 混合 | w={w:.2f} | 覆盖率: {covered}/{num_dancers}")
    elif prev_week_estimates is None or len(prev_week_estimates) == 0:
        # 第1周：来源是热度（可能退化为均匀）
        covered = int(np.sum(has_source_coverage)) if has_source_coverage is not None else 0
        if verbose:
            print(f"    [先验] 第1周: 惯性(热度) + 路人(均匀) 混合 | w={w:.2f} | 覆盖率: {covered}/{num_dancers}")
    else:
        covered = int(np.sum(has_source_coverage)) if has_source_coverage is not None else 0
        if verbose:
            print(f"    [先验] 惯性(上周后验) + 路人(均匀) 混合 | w={w:.2f} | 覆盖率: {covered}/{num_dancers}")

    return alpha


def run_simulation_percent_mode(season_num, week_num, df, prev_week_estimates=None,
                                concentration=50.0, iterations=100000, use_bayesian=True,
                                popularity_data=None, inertia_weight=None, verbose=True):
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
        use_bayesian: 是否使用贝叶斯时序一致性（默认 True）
    
    返回:
        (result_df, current_week_estimates, has_elimination): 
            结果 DataFrame、当周估计字典、本周是否有淘汰
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
        return None, None, False
    
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
    # Step 3: 确定实际淘汰者（合并查找名字和索引）
    # ---------------------------------------------------------
    actual_loser_idx, actual_eliminated_name = None, None
    for i, (_, row) in enumerate(active_df.iterrows()):
        if extract_eliminated_week(row.get('results')) == week_num:
            actual_eliminated_name = row['celebrity_name']
            actual_loser_idx = i
            break
    
    has_elimination = actual_loser_idx is not None
    
    if verbose:
        if not has_elimination:
            print(f"Season {season_num} Week {week_num} [{MODE_DISPLAY[mode]}]: 无淘汰记录，所有样本均有效。")
        else:
            print(f"--- Processing Season {season_num} Week {week_num} [{MODE_DISPLAY[mode]}] (Target: {actual_eliminated_name}) ---")
    
    # ---------------------------------------------------------
    # Step 4: 构造 Dirichlet 先验参数 α
    # ---------------------------------------------------------
    w = INERTIA_WEIGHT if inertia_weight is None else inertia_weight
    force_first_week_prior = bool(use_bayesian) and bool(BAYESIAN_STATIC_WEEK1_PRIOR)
    prior_prev = None if force_first_week_prior else prev_week_estimates
    alpha = construct_dirichlet_prior(
        prior_prev,
        names,
        concentration,
        use_bayesian=use_bayesian,
        popularity_data=popularity_data,
        season_num=season_num,
        inertia_weight=w,
        verbose=verbose,
        force_first_week_prior=force_first_week_prior,
    )
    
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
        if verbose:
            print(f"    警告：没有找到符合历史结果的有效样本（concentration={concentration}）。")
            print(f"    尝试放宽约束或增加迭代次数。")
        return None, None, has_elimination
    
    acceptance_rate = valid_count / iterations * 100.0
    if verbose:
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
    
    return pd.DataFrame(results), current_week_estimates, has_elimination


def _build_eval_schedule(df, weeks_to_simulate, seasons):
    """与 stable 主循环一致：每个 season 从 week1 开始，直到 active_count<=1 便停止。"""
    schedule_by_season = {}
    total_tasks = 0
    for season in seasons:
        weeks = []
        for week in weeks_to_simulate:
            active_count = count_active_dancers(season, week, df)
            if active_count <= 1:
                break
            weeks.append(week)
        schedule_by_season[season] = weeks
        total_tasks += len(weeks)
    return schedule_by_season, total_tasks


def evaluate_inertia_weight(
    df,
    popularity_df,
    inertia_weight,
    weeks_to_simulate,
    seasons=None,
    concentration=50.0,
    iterations=20000,
    schedule_by_season=None,
    progress_bar=None,
):
    """评估某个 inertia_weight 的“命中/没命中”统计，口径与 stable 版本完全一致。

    stable 口径：
    - result_df is None  <=>  valid_count==0  <=> no-hit
    - 统计 overall 与 elimination weeks 两类。

    返回:
        (stats_dict)
    """
    seasons = TARGET_SEASONS if seasons is None else seasons

    weeks_attempted = 0
    weeks_no_hit = 0
    elim_weeks_attempted = 0
    elim_weeks_no_hit = 0

    # 额外统计（用于 CSV 输出）：三种赛制的平均接受率与平均标准差
    # 注意：这里的“平均”按【选手】加权，而不是按周：
    # - 平均接受率：对每个 dancer-week 行加权（同一周每行 Acceptance_Rate 相同，相当于按当周活跃人数加权）
    #   * no-hit 周：按 0 计入，但分母仍加上该周活跃人数（使 no-hit 周对均值有惩罚）
    # - 平均标准差：对所有有结果的 dancer-week 行的 Est_Fan_Percent_Std 直接取均值（no-hit 周无 posterior，无法计入）
    mode_keys = ['rank_strict', 'percentage', 'rank_bottom2']
    weeks_attempted_by_mode = {k: 0 for k in mode_keys}
    weeks_no_hit_by_mode = {k: 0 for k in mode_keys}
    # dancer-weighted sums
    acc_weighted_sum_by_mode = {k: 0.0 for k in mode_keys}
    acc_weighted_count_by_mode = {k: 0 for k in mode_keys}

    std_sum_by_mode = {k: 0.0 for k in mode_keys}
    std_count_by_mode = {k: 0 for k in mode_keys}

    for season in seasons:
        prev_estimates = None

        season_mode = get_season_mode(season)
        if season_mode not in mode_keys:
            season_mode = 'percentage'

        if schedule_by_season is not None and season in schedule_by_season:
            weeks_iter = schedule_by_season[season]
        else:
            weeks_iter = weeks_to_simulate

        for week in weeks_iter:
            if schedule_by_season is None:
                active_count = count_active_dancers(season, week, df)
                if active_count <= 1:
                    break

            result_df, curr_estimates, had_elimination = run_simulation_percent_mode(
                season_num=season,
                week_num=week,
                df=df,
                prev_week_estimates=prev_estimates,
                concentration=concentration,
                iterations=iterations,
                use_bayesian=True,
                popularity_data=popularity_df,
                inertia_weight=inertia_weight,
                verbose=False,
            )

            # 与 stable 一致：先计数 attempted，再根据 result_df 是否为 None 计 no-hit。
            weeks_attempted += 1
            weeks_attempted_by_mode[season_mode] += 1
            if had_elimination:
                elim_weeks_attempted += 1

            if result_df is None:
                weeks_no_hit += 1
                weeks_no_hit_by_mode[season_mode] += 1
                if had_elimination:
                    elim_weeks_no_hit += 1
                # no-hit 周：没有有效样本，接受率按 0 计入平均。
                # 这里按“选手”加权：分母增加该周活跃选手数。
                try:
                    n_active = int(count_active_dancers(season, week, df))
                except Exception:
                    n_active = 0
                if n_active > 0:
                    acc_weighted_sum_by_mode[season_mode] += 0.0
                    acc_weighted_count_by_mode[season_mode] += n_active
            else:
                # 有结果：
                # - acceptance_rate 在每个 dancer 行重复，取第一行即可
                # - 按“选手”加权：Acceptance_Rate * 行数
                # - std 按“选手”逐行累加
                try:
                    acc = float(result_df.iloc[0]['Acceptance_Rate'])
                except Exception:
                    acc = 0.0

                try:
                    n_rows = int(len(result_df))
                except Exception:
                    n_rows = 0
                if n_rows > 0:
                    if np.isfinite(acc):
                        acc_weighted_sum_by_mode[season_mode] += acc * n_rows
                        acc_weighted_count_by_mode[season_mode] += n_rows

                    if 'Est_Fan_Percent_Std' in result_df.columns:
                        try:
                            std_vals = pd.to_numeric(result_df['Est_Fan_Percent_Std'], errors='coerce')
                            std_vals = std_vals[np.isfinite(std_vals)]
                            if len(std_vals) > 0:
                                std_sum_by_mode[season_mode] += float(std_vals.sum())
                                std_count_by_mode[season_mode] += int(len(std_vals))
                        except Exception:
                            pass

            # 与 stable 一致：只在“淘汰周且有结果且非静态先验”时更新先验
            if (had_elimination
                and (result_df is not None)
                and (not BAYESIAN_STATIC_WEEK1_PRIOR)):
                prev_estimates = curr_estimates

            if progress_bar is not None:
                progress_bar.update(1)

    hit_rate_overall = 0.0
    if weeks_attempted > 0:
        hit_rate_overall = 100.0 * (1.0 - (weeks_no_hit / weeks_attempted))

    hit_rate_elim = 0.0
    if elim_weeks_attempted > 0:
        hit_rate_elim = 100.0 * (1.0 - (elim_weeks_no_hit / elim_weeks_attempted))

    # 平均接受率：按“选手”加权（dancer-week 行）。no-hit 周按 0 计入，但分母仍计入活跃人数。
    avg_acc_by_mode = {
        k: (acc_weighted_sum_by_mode[k] / acc_weighted_count_by_mode[k] if acc_weighted_count_by_mode[k] > 0 else 0.0)
        for k in mode_keys
    }

    # 平均标准差：按“选手”逐行平均（no-hit 周没有 posterior，无法计入）
    avg_std_by_mode = {
        k: (std_sum_by_mode[k] / std_count_by_mode[k] if std_count_by_mode[k] > 0 else 0.0)
        for k in mode_keys
    }

    return {
        'weeks_attempted': int(weeks_attempted),
        'weeks_no_hit': int(weeks_no_hit),
        'elim_weeks_attempted': int(elim_weeks_attempted),
        'elim_weeks_no_hit': int(elim_weeks_no_hit),
        'hit_rate_overall': float(hit_rate_overall),
        'hit_rate_elim': float(hit_rate_elim),

        # per-mode
        'weeks_attempted_by_mode': weeks_attempted_by_mode,
        'weeks_no_hit_by_mode': weeks_no_hit_by_mode,
        'avg_acceptance_by_mode': avg_acc_by_mode,
        'avg_std_by_mode': avg_std_by_mode,
    }


def grid_search_inertia_weight(
    df,
    popularity_df,
    weeks_to_simulate,
    weights,
    concentration=50.0,
    iterations=20000,
    seasons=None,
    output_csv='inertia_weight_grid_search.csv',
    show_progress=True,
):
    """对 inertia_weight 做网格搜索（专用）。

    输出结果与 stable 口径一致：基于 no-hit 统计。
    """
    rows = []

    seasons = TARGET_SEASONS if seasons is None else seasons
    schedule_by_season, total_tasks = _build_eval_schedule(df, weeks_to_simulate, seasons)
    tasks_per_weight = total_tasks

    outer_iter = weights
    if show_progress and tqdm is not None:
        outer_iter = tqdm(list(weights), desc='Grid Search (weights)', unit='w')

    for w in outer_iter:
        inner_bar = None
        if show_progress and tqdm is not None:
            inner_bar = tqdm(total=tasks_per_weight, desc=f'Evaluating w={float(w):.3f}', unit='week', leave=False)

        stats = evaluate_inertia_weight(
            df=df,
            popularity_df=popularity_df,
            inertia_weight=float(w),
            weeks_to_simulate=weeks_to_simulate,
            seasons=seasons,
            concentration=concentration,
            iterations=iterations,
            schedule_by_season=schedule_by_season,
            progress_bar=inner_bar,
        )

        if inner_bar is not None:
            inner_bar.close()

        avg_acc_by_mode = stats.get('avg_acceptance_by_mode', {})
        avg_std_by_mode = stats.get('avg_std_by_mode', {})
        weeks_attempted_by_mode = stats.get('weeks_attempted_by_mode', {})
        weeks_no_hit_by_mode = stats.get('weeks_no_hit_by_mode', {})

        rows.append({
            'inertia_weight': float(w),
            # 与 stable 输出一致的 no-hit 统计
            'weeks_attempted': stats['weeks_attempted'],
            'weeks_no_hit': stats['weeks_no_hit'],
            'elim_weeks_attempted': stats['elim_weeks_attempted'],
            'elim_weeks_no_hit': stats['elim_weeks_no_hit'],

            # 便于排序/直观展示（由 stable 统计推导得来）
            'hit_rate_overall_%': round(stats['hit_rate_overall'], 6),
            'hit_rate_elim_%': round(stats['hit_rate_elim'], 6),

            # 三种赛制：平均接受率（%）与平均标准差（%）
            'avg_acceptance_rate_rank_strict': round(float(avg_acc_by_mode.get('rank_strict', 0.0)), 6),
            'avg_acceptance_rate_percentage': round(float(avg_acc_by_mode.get('percentage', 0.0)), 6),
            'avg_acceptance_rate_rank_bottom2': round(float(avg_acc_by_mode.get('rank_bottom2', 0.0)), 6),

            'avg_std_rank_strict': round(float(avg_std_by_mode.get('rank_strict', 0.0)), 6),
            'avg_std_percentage': round(float(avg_std_by_mode.get('percentage', 0.0)), 6),
            'avg_std_rank_bottom2': round(float(avg_std_by_mode.get('rank_bottom2', 0.0)), 6),

            # 三种赛制：周数（便于核对）
            'weeks_attempted_rank_strict': int(weeks_attempted_by_mode.get('rank_strict', 0)),
            'weeks_attempted_percentage': int(weeks_attempted_by_mode.get('percentage', 0)),
            'weeks_attempted_rank_bottom2': int(weeks_attempted_by_mode.get('rank_bottom2', 0)),
            'weeks_no_hit_rank_strict': int(weeks_no_hit_by_mode.get('rank_strict', 0)),
            'weeks_no_hit_percentage': int(weeks_no_hit_by_mode.get('percentage', 0)),
            'weeks_no_hit_rank_bottom2': int(weeks_no_hit_by_mode.get('rank_bottom2', 0)),

            'iterations': int(iterations),
            'concentration': float(concentration),
        })

    result = pd.DataFrame(rows)
    # 默认按“淘汰周命中率”优先排序，更符合实际约束周的有效性
    result = result.sort_values(['hit_rate_elim_%', 'hit_rate_overall_%'], ascending=False).reset_index(drop=True)
    result.to_csv(output_csv, index=False)

    best = result.iloc[0].to_dict() if len(result) > 0 else None
    return result, best


def _parse_args():
    p = argparse.ArgumentParser(description='Grid search inertia_weight with stable-style hit metric (no-hit weeks).')
    p.add_argument('--grid-start', type=float, default=0.0)
    p.add_argument('--grid-end', type=float, default=1.0)
    p.add_argument('--grid-step', type=float, default=0.05)
    p.add_argument('--grid-iter', type=int, default=20000, help='Monte Carlo iterations per week during evaluation')
    p.add_argument('--concentration', type=float, default=CONCENTRATION)
    p.add_argument('--out', type=str, default='inertia_weight_grid_search.csv')
    p.add_argument('--no-progress', action='store_true')
    return p.parse_args()


def main():
    args = _parse_args()

    # 加载数据
    df = pd.read_csv('2026_MCM_Problem_C_Data.csv')

    # 加载热度系数数据
    try:
        popularity_df = pd.read_csv(POPULARITY_FILE)
        print(f"已加载热度系数数据: {len(popularity_df)} 条记录")
    except FileNotFoundError:
        popularity_df = None
        print(f"警告: 未找到热度系数文件 {POPULARITY_FILE}，将使用均匀先验")

    # 数据清洗：将 N/A 转为 0，并将分数转为浮点数
    score_cols = [c for c in df.columns if 'score' in c]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    weeks_to_simulate = get_available_weeks(df)

    # 生成包含 end 的网格
    weights = np.arange(args.grid_start, args.grid_end + 1e-12, args.grid_step)

    print("=" * 70)
    print("Grid Search: INERTIA_WEIGHT (stable-style hit metric)")
    print(f"Grid: start={args.grid_start}, end={args.grid_end}, step={args.grid_step} | points={len(weights)}")
    print(f"Eval: iterations={args.grid_iter}, concentration={args.concentration}")
    print("Metric: no-hit weeks (result_df is None) and no-hit elimination weeks")
    print("=" * 70)

    result_df, best = grid_search_inertia_weight(
        df=df,
        popularity_df=popularity_df,
        weeks_to_simulate=weeks_to_simulate,
        weights=weights,
        concentration=float(args.concentration),
        iterations=int(args.grid_iter),
        seasons=TARGET_SEASONS,
        output_csv=args.out,
        show_progress=(not args.no_progress),
    )

    if best is not None:
        print("\nTop 10 candidates:")
        print(result_df.head(10))
        print("\nBest INERTIA_WEIGHT:")
        print(best)
        print(f"\nGrid search results saved to: {args.out}")
    else:
        print("No grid search results generated.")


if __name__ == '__main__':
    main()
