import pandas as pd
import numpy as np

def simulate_four_voting_systems():
    # 1. 加载数据
    file_path = '2026_MCM_Problem_C_Data_with_bayesian_weekly.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return

    # 目标选手列表
    targets = [
        ("Jerry Rice", 2),
        ("Billy Ray Cyrus", 4),
        ("Bristol Palin", 11),
        ("Bobby Bones", 27)
    ]

    print("="*120)
    print(" 四种计票机制下的争议选手命运模拟 ")
    print("="*120)
    print("图例说明：")
    print("  SAFE:        安全晋级")
    print("  ELIM (Low):  因总分/排名垫底被直接淘汰")
    print("  ELIM (Judge):进入Bottom 2后，因裁判分低于对手而被裁判淘汰")
    print("  SAVED:       进入Bottom 2后，因裁判分高于对手而被裁判保送")
    print("-" * 120)

    # 2. 数据清洗
    for w in range(1, 13):
        for j in range(1, 5):
            c = f'week{w}_judge{j}_score'
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 3. 核心模拟逻辑
    for target_name, season in targets:
        print(f"\n>>> 选手: {target_name} (Season {season})")
        # 表头
        print(f"{'Week':<4} | {'J_Score':<7} | {'1. Pure Percent':<18} | {'2. Pure Rank':<18} | {'3. Pct + Judge Save':<20} | {'4. Rank + Judge Save':<20}")
        print("-" * 115)
        
        season_data = df[df['season'] == season].copy()
        
        for week in range(1, 12):
            # 准备数据列
            fan_col = f'week{week}_Est_FanPct_Mean_Bayesian'
            judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5) if f'week{week}_judge{j}_score' in df.columns]

            if fan_col not in df.columns or not judge_cols:
                continue

            # 获取当周数据
            week_df = season_data.copy()
            week_df['total_judge'] = week_df[judge_cols].sum(axis=1)
            # 筛选活跃选手
            active = week_df[(week_df['total_judge'] > 0) & (week_df[fan_col] > 0)].copy()
            
            if len(active) < 3: # 决赛周跳过
                continue
            
            if target_name not in active['celebrity_name'].values:
                break # 选手已不在比赛中

            # === 计算指标 ===
            # 百分比法指标
            total_pts = active['total_judge'].sum()
            active['pct_j'] = (active['total_judge'] / total_pts) * 100
            active['total_pct'] = active['pct_j'] + active[fan_col]
            
            # 排名法指标
            active['rank_j'] = active['total_judge'].rank(ascending=False, method='min')
            active['rank_f'] = active[fan_col].rank(ascending=False, method='min')
            active['sum_rank'] = active['rank_j'] + active['rank_f']

            # 获取目标选手的得分用于后续判断
            t_row = active[active['celebrity_name'] == target_name].iloc[0]
            t_score = t_row['total_judge']

            # === 模拟四种机制 ===
            results = []

            # 1. Pure Percent (最低分淘汰)
            # 排序：总百分比从小到大
            sorted_pct = active.sort_values(by='total_pct', ascending=True)
            eliminated_pure_pct = sorted_pct.iloc[0]['celebrity_name']
            res1 = "ELIM (Low)" if eliminated_pure_pct == target_name else "SAFE"

            # 2. Pure Rank (最高Rank淘汰)
            # 排序：总Rank从大到小，若平局则裁判分从低到高
            sorted_rank = active.sort_values(by=['sum_rank', 'total_judge'], ascending=[False, True])
            eliminated_pure_rank = sorted_rank.iloc[0]['celebrity_name']
            res2 = "ELIM (Low)" if eliminated_pure_rank == target_name else "SAFE"

            # 3. Percent + Judge Save (Bottom 2 by Pct -> Judge Decision)
            b2_pct_df = sorted_pct.head(2) # 取倒数两名
            if target_name in b2_pct_df['celebrity_name'].values:
                # 找出对手
                opponent = b2_pct_df[b2_pct_df['celebrity_name'] != target_name].iloc[0]
                if t_score < opponent['total_judge']:
                    res3 = "ELIM (Judge)"
                elif t_score > opponent['total_judge']:
                    res3 = "SAVED"
                else:
                    res3 = "TIE (Danger)"
            else:
                res3 = "SAFE"

            # 4. Rank + Judge Save (Bottom 2 by Rank -> Judge Decision)
            b2_rank_df = sorted_rank.head(2) # 取倒数两名
            if target_name in b2_rank_df['celebrity_name'].values:
                # 找出对手
                opponent = b2_rank_df[b2_rank_df['celebrity_name'] != target_name].iloc[0]
                if t_score < opponent['total_judge']:
                    res4 = "ELIM (Judge)"
                elif t_score > opponent['total_judge']:
                    res4 = "SAVED"
                else:
                    res4 = "TIE (Danger)"
            else:
                res4 = "SAFE"

            # 打印该周结果
            print(f"{week:<4} | {t_score:<7} | {res1:<18} | {res2:<18} | {res3:<20} | {res4:<20}")

    print("\n" + "="*120)
    print("分析完成。")

if __name__ == "__main__":
    simulate_four_voting_systems()