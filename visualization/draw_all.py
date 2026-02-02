import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据加载与预处理
# ==========================================
# 读取数据
filename = '2026_MCM_Problem_C_Data_with_bayesian_weekly.csv'
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: 找不到文件 {filename}，请确保文件在当前目录下。")
    # 为了演示代码可运行，这里如果找不到文件会报错退出
    exit()

# 确保 placement 是数值型
df['placement'] = pd.to_numeric(df['placement'], errors='coerce')

# 提取周数范围（自动检测最大周数）
week_cols = [c for c in df.columns if 'week' in c and 'judge' in c]
max_week_num = 0
for c in week_cols:
    try:
        # 提取 "week" 和 "_" 之间的数字
        num = int(c.split('week')[1].split('_')[0])
        if num > max_week_num:
            max_week_num = num
    except:
        pass
weeks = list(range(1, max_week_num + 1))

# ==========================================
# 2. 数据聚合 (计算每个选手的生涯平均分)
# ==========================================
summary_stats = []

for idx, row in df.iterrows():
    p_judge_scores = []
    p_fan_votes = []
    
    for w in weeks:
        # A. 获取每周裁判平均分
        j_cols = [c for c in df.columns if f'week{w}_judge' in c and 'score' in c]
        scores = pd.to_numeric(row[j_cols], errors='coerce')
        
        if scores.sum() > 0: # 只要有分就算存活
            avg_j = scores.mean()
            p_judge_scores.append(avg_j)
            
        # B. 获取每周粉丝投票估算
        fan_col = f'week{w}_Est_FanPct_Mean_Bayesian'
        if fan_col in df.columns:
            val = pd.to_numeric(row[fan_col], errors='coerce')
            if not np.isnan(val) and val > 0:
                p_fan_votes.append(val)
    
    # 只有当选手至少有一周数据时才统计
    if p_judge_scores:
        avg_judge_total = np.mean(p_judge_scores)
        # 如果没有粉丝数据（极少数情况），给一个默认值或跳过，这里取均值
        avg_fan_total = np.mean(p_fan_votes) if p_fan_votes else 0
        weeks_survived = len(p_judge_scores)
        
        summary_stats.append({
            'Name': row['celebrity_name'],
            'Season': row['season'],
            'Placement': row['placement'],
            'Avg_Judge_Score': avg_judge_total,
            'Avg_Fan_Vote': avg_fan_total,
            'Weeks_Survived': weeks_survived,
            'Is_Winner': row['placement'] == 1
        })

df_summary = pd.DataFrame(summary_stats)

# ==========================================
# 3. 绘制美学气泡象限图
# ==========================================

# 设置画布
plt.figure(figsize=(12, 9), dpi=300) # 高分辨率
sns.set_theme(style="whitegrid", font_scale=1.1)

# 定义象限分割线（基于中位数）
mid_x = df_summary['Avg_Judge_Score'].median()
mid_y = df_summary['Avg_Fan_Vote'].median()

# --- 步骤 A: 绘制背景象限和区域标注 ---
plt.axvline(x=mid_x, color='gray', linestyle=':', alpha=0.6, zorder=0)
plt.axhline(y=mid_y, color='gray', linestyle=':', alpha=0.6, zorder=0)

# 添加象限含义文字 (使用相对坐标，更整洁)
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
plt.text(9.8, df_summary['Avg_Fan_Vote'].max(), 'Top Contenders\n(Skill + Popularity)', 
         fontsize=10, color='green', ha='right', va='top', bbox=props)
plt.text(df_summary['Avg_Judge_Score'].min(), df_summary['Avg_Fan_Vote'].max(), 'Fan Favorites\n(Popular but Less Skilled)', 
         fontsize=10, color='blue', ha='left', va='top', bbox=props)
plt.text(9.8, df_summary['Avg_Fan_Vote'].min(), 'Shock Eliminations\n(Skilled but Unpopular)', 
         fontsize=10, color='orange', ha='right', va='bottom', bbox=props)

# --- 步骤 B: 绘制非冠军选手 (圆形气泡) ---
# 过滤非冠军
non_winners = df_summary[df_summary['Is_Winner'] == False]

# 颜色映射：使用 viridis_r (倒序)，这样排名靠前(数值小)是黄色/亮色，排名靠后是紫色/深色
scatter = sns.scatterplot(
    data=non_winners,
    x='Avg_Judge_Score',
    y='Avg_Fan_Vote',
    size='Weeks_Survived',
    hue='Placement',
    palette='viridis_r', 
    sizes=(30, 400), # 气泡大小范围
    alpha=0.6,       # 半透明
    edgecolor='white',
    linewidth=0.5,
    legend='brief',
    zorder=2
)

# --- 步骤 C: 绘制冠军选手 (金色五角星) ---
# 过滤冠军
winners = df_summary[df_summary['Is_Winner'] == True]

# 单独绘制冠军，使用星形标记
plt.scatter(
    x=winners['Avg_Judge_Score'],
    y=winners['Avg_Fan_Vote'],
    s=350,              # 星星的大小
    c='gold',           # 金色填充
    marker='*',         # 星形
    edgecolors='black', # 黑色边框
    linewidths=0.8,
    label='Season Winners',
    zorder=3            # 保证显示在最上层
)

# --- 步骤 D: 图表装饰与标签 ---
plt.title("The Dancing Universe: Technical Merit vs. Fan Popularity", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Average Judge Score (Technical Ability)", fontsize=13)
plt.ylabel("Average Estimated Fan Vote % (Popularity)", fontsize=13)

# 处理图例
# 获取 Seaborn 生成的句柄
h, l = scatter.get_legend_handles_labels()
# 添加 "Winner" 的图例项
from matplotlib.lines import Line2D
winner_proxy = Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                      markeredgecolor='black', markersize=15, label='Season Winner')
# 重新组合图例 (保留Placement的一部分，加上Winner)
# 这里的切片是为了去掉Seaborn图例中一些多余的标题
final_handles = [winner_proxy] + h[1:6] + h[-4:] 
final_labels = ['Season Winner'] + l[1:6] + l[-4:]

plt.legend(final_handles, final_labels, title="Placement & Longevity", 
           bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)

plt.tight_layout()

# ==========================================
# 4. 保存结果
# ==========================================
save_path = 'dancing_universe_analysis.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图表已成功保存为: {save_path}")

plt.show()