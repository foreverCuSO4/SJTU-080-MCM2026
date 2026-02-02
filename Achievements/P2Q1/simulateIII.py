import pandas as pd
import numpy as np

def solve_voting_comparison():
    # 1. 加载数据
    # 请确保csv文件在当前目录下，或者修改为正确路径
    file_path = '2026_MCM_Problem_C_Data_with_bayesian_weekly.csv'
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}。请确保文件存在。")
        return

    # 2. 数据预处理
    # 将裁判分数转换为数值，无法解析的设为0
    judge_cols = []
    for w in range(1, 12): # 假设最多11周
        for j in range(1, 5): # 最多4位裁判
            col_name = f'week{w}_judge{j}_score'
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
                judge_cols.append(col_name)

    # 结果存储列表
    comparison_results = []
    
    # 获取所有赛季
    seasons = sorted(df['season'].unique())

    for season in seasons:
        season_data = df[df['season'] == season]
        
        # 遍历该赛季的每一周
        # 注意：通常只有在前一周未被淘汰的人才参与。
        # 这里我们通过判断 Total Judge Score > 0 来确定选手本周是否参赛
        for week in range(1, 12):
            # 构建列名
            current_judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5) if f'week{week}_judge{j}_score' in df.columns]
            fan_pct_col = f'week{week}_Est_FanPct_Mean_Bayesian'
            
            # 检查列是否存在
            if not current_judge_cols or fan_pct_col not in df.columns:
                continue
            
            # 计算该周裁判总分
            # 使用 .copy() 避免 SettingWithCopyWarning
            week_df = season_data.copy()
            week_df['total_judge_score'] = week_df[current_judge_cols].sum(axis=1)
            
            # 筛选出本周参赛的选手（裁判总分 > 0 且 粉丝数据存在）
            active_contestants = week_df[
                (week_df['total_judge_score'] > 0) & 
                (week_df[fan_pct_col].notna()) & 
                (week_df[fan_pct_col] > 0)
            ].copy()
            
            # 如果参赛人数少于2人（通常是决赛或数据缺失），跳过
            if len(active_contestants) < 2:
                continue

            # ==========================
            # 方法 A: 排名法 (Rank Method)
            # ==========================
            # 逻辑：
            # 1. 裁判分越高，排名数字越小 (1st place = Best)
            # 2. 粉丝票越高，排名数字越小 (1st place = Best)
            # 3. 相加 Ranks。
            # 4. Sum 越大，表示表现越差，越容易被淘汰。
            
            # rank(method='min') 意味着如果有平局，都取最好的名次（例如两个第一名，下一个是第三名）
            # ascending=False: 分数高者排在前面(Rank 1)
            active_contestants['rank_judge'] = active_contestants['total_judge_score'].rank(ascending=False, method='min')
            active_contestants['rank_fan'] = active_contestants[fan_pct_col].rank(ascending=False, method='min')
            active_contestants['sum_ranks'] = active_contestants['rank_judge'] + active_contestants['rank_fan']
            
            # 找出 Sum Ranks 最大的（表现最差）。
            # 平局处理：通常规则是“裁判分更低者淘汰”。
            # 我们先按 Sum Ranks 降序排，再按 Judge Score 升序排（分数低的优先被淘汰）
            rank_sorted = active_contestants.sort_values(by=['sum_ranks', 'total_judge_score'], ascending=[False, True])
            eliminated_by_rank = rank_sorted.iloc[0] # 取第一个为淘汰者

            # ==========================
            # 方法 B: 百分比法 (Percentage Method)
            # ==========================
            # 逻辑：
            # 1. 计算裁判分数占比
            # 2. 加上粉丝分数占比 (已在 Est_FanPct 列中)
            # 3. 总分最低者淘汰
            
            total_judges_points_week = active_contestants['total_judge_score'].sum()
            active_contestants['pct_judge'] = (active_contestants['total_judge_score'] / total_judges_points_week) * 100
            active_contestants['pct_total'] = active_contestants['pct_judge'] + active_contestants[fan_pct_col]
            
            # 找出 pct_total 最小的
            pct_sorted = active_contestants.sort_values(by=['pct_total'], ascending=[True])
            eliminated_by_pct = pct_sorted.iloc[0]

            # ==========================
            # 比较与记录
            # ==========================
            is_different = eliminated_by_rank['celebrity_name'] != eliminated_by_pct['celebrity_name']
            
            # 只有当两种方法产生不同结果时，分析才有意义，但我们记录所有数据以便统计
            comparison_results.append({
                'season': season,
                'week': week,
                'contestants_count': len(active_contestants),
                # Rank Method Result
                'rank_elim_name': eliminated_by_rank['celebrity_name'],
                'rank_elim_judge_score': eliminated_by_rank['total_judge_score'],
                'rank_elim_fan_pct': eliminated_by_rank[fan_pct_col],
                # Pct Method Result
                'pct_elim_name': eliminated_by_pct['celebrity_name'],
                'pct_elim_judge_score': eliminated_by_pct['total_judge_score'],
                'pct_elim_fan_pct': eliminated_by_pct[fan_pct_col],
                # Diff Flag
                'different_outcome': is_different
            })

    # 3. 分析结果
    results_df = pd.DataFrame(comparison_results)
    
    diff_df = results_df[results_df['different_outcome'] == True]
    
    print("="*60)
    print(" 投票方法对比分析 (排名法 vs 百分比法) ")
    print("="*60)
    print(f"总模拟周次: {len(results_df)}")
    print(f"结果不一致的周次: {len(diff_df)} ({len(diff_df)/len(results_df)*100:.2f}%)")
    print("-" * 60)
    
    if len(diff_df) > 0:
        print("\n[差异案例详细分析]")
        print("以下列出了两种方法导致淘汰人选不同的情况：")
        print(f"{'Season':<6} | {'Week':<4} | {'Rank法淘汰 (粉丝%)':<25} | {'Percent法淘汰 (粉丝%)':<25}")
        print("-" * 75)
        
        pct_method_saved_fan_favorite_count = 0
        rank_method_saved_fan_favorite_count = 0

        for _, row in diff_df.iterrows():
            r_name = row['rank_elim_name']
            r_fan = row['rank_elim_fan_pct']
            p_name = row['pct_elim_name']
            p_fan = row['pct_elim_fan_pct']
            
            print(f"S{row['season']:<5} | W{row['week']:<3} | {r_name} ({r_fan:.1f}%)      | {p_name} ({p_fan:.1f}%)")
            
            # 分析：哪种方法淘汰了粉丝支持率更高的人？
            # 如果 Rank法淘汰了 A (粉丝 10%)，Percent法淘汰了 B (粉丝 5%)
            # 说明 Percent法 拯救了 A。
            # 同时也说明 Rank法 对粉丝更不友好（淘汰了粉丝多的人）。
            
            if r_fan > p_fan:
                # Rank法淘汰了粉丝更多的人 -> Percent法保护了粉丝更多的人
                pct_method_saved_fan_favorite_count += 1
            elif p_fan > r_fan:
                # Percent法淘汰了粉丝更多的人 -> Rank法保护了粉丝更多的人
                rank_method_saved_fan_favorite_count += 1

        print("-" * 75)
        print("\n[结论：哪种方法更倾向于粉丝？]")
        print(f"在 {len(diff_df)} 次结果不一致的情况中：")
        print(f"1. 百分比法 (Percent) '拯救'了粉丝票更高选手的次数: {pct_method_saved_fan_favorite_count}")
        print(f"2. 排名法 (Rank)    '拯救'了粉丝票更高选手的次数: {rank_method_saved_fan_favorite_count}")
        
        if pct_method_saved_fan_favorite_count > rank_method_saved_fan_favorite_count:
            print("\n>>> 分析结论: 百分比法 (Percentage Method) 似乎更有利于粉丝投票。")
            print("    原因：在冲突情况下，该方法更常保留住那些裁判分低但拥有较高粉丝占比的选手。")
        elif rank_method_saved_fan_favorite_count > pct_method_saved_fan_favorite_count:
            print("\n>>> 分析结论: 排名法 (Rank Method) 似乎更有利于粉丝投票。")
        else:
            print("\n>>> 分析结论: 两种方法对粉丝投票权重的偏向性没有明显差异。")
            
    else:
        print("在所有模拟数据中，两种方法产生的结果完全一致。")

    # 可选：将差异数据保存为CSV以供报告使用
    # diff_df.to_csv('method_comparison_diff.csv', index=False)

if __name__ == "__main__":
    solve_voting_comparison()