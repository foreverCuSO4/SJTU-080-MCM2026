import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

# ==========================================
# 0. 配置选项
# ==========================================
# 选项：设置为 True 则只绘制 Uniform Model 的点；设置为 False 则绘制两者对比
ONLY_SHOW_UNIFORM = False

# 默认输出尺寸（英寸），16:9
FIGSIZE_INCHES = (16, 12)

# 文件名
filename = 'monte_carlo_results_1M_all_hit_0201.csv'

# ==========================================
# 1. 读取数据与预处理
# ==========================================
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"错误: 找不到文件 {filename}")
    exit()

# 将宽表转为适合绘图的长表 (Tidy Data)
data_long = []
for _, row in df.iterrows():
    # 提取 Bayesian 数据
    data_long.append({
        'Std': row['Est_Fan%_Std_Bayesian'],
        'Acceptance_Rate': row['Acceptance_Rate%_Bayesian'],
        'Method': 'Bayesian Model'
    })
    # 提取 Uniform 数据
    data_long.append({
        'Std': row['Est_Fan%_Std_Uniform'],
        'Acceptance_Rate': row['Acceptance_Rate%_Uniform'],
        'Method': 'Uniform Model'
    })

df_plot = pd.DataFrame(data_long)

# 将关键列转为数值（更鲁棒：遇到非数值会变成 NaN，后续统一 drop）
df_plot['Acceptance_Rate'] = pd.to_numeric(df_plot['Acceptance_Rate'], errors='coerce')
df_plot['Std'] = pd.to_numeric(df_plot['Std'], errors='coerce')

# ==========================================
# 1.5 清洗数据：数值化 + 去缺失
# ==========================================
# 注意：均值统计口径使用“清洗后但不剔除 100% 接受率点”的数据；
#       而绘图时仍不显示 100% 的点（避免图面被边界点占据）。
df_plot = df_plot.dropna(subset=['Acceptance_Rate', 'Std'])

# 用于均值统计：保留 100% 接受率点
df_for_mean = df_plot.copy()

# 用于绘图显示：过滤掉接受率为 100%（含浮点近似）的点
df_plot = df_plot[~np.isclose(df_plot['Acceptance_Rate'].to_numpy(dtype=float), 100.0, atol=1e-9)]

# ==========================================
# 2. 根据选项过滤数据
# ==========================================
if ONLY_SHOW_UNIFORM:
    df_plot = df_plot[df_plot['Method'] == 'Uniform Model']
    print("模式已启用：仅绘制 Uniform Model 数据")

# ==========================================
# 2.5 计算并准备标注：两种模型的均值（不剔除 100% 点）
# ==========================================
means_df = (
    (df_for_mean if not ONLY_SHOW_UNIFORM else df_for_mean[df_for_mean['Method'] == 'Uniform Model'])
    .groupby('Method', as_index=False)[['Acceptance_Rate', 'Std']]
    .mean()
)

# ==========================================
# 3. 设置绘图参数
# ==========================================
sns.set_theme(style="white", font_scale=1.2)

# 定义颜色：Bayesian使用深青色，Uniform使用珊瑚色
# 注意：即使过滤了数据，保留字典也没问题，Seaborn会自动匹配
custom_palette = {'Bayesian Model': '#2E86AB', 'Uniform Model': '#F24236'}

# ==========================================
# 4. 创建联合分布图 (JointGrid)
# ==========================================
# 创建画布
g = sns.JointGrid(data=df_plot, x="Acceptance_Rate", y="Std", hue="Method", 
                  height=8, ratio=4, palette=custom_palette)

# 将输出图像尺寸固定到 16:9（更适合论文/幻灯片）
g.fig.set_size_inches(*FIGSIZE_INCHES)

# --- B. 绘制中心散点图 (密度可视化核心) ---
# 修改说明：
# 1. alpha=0.3: 透明度设低，这样点叠加时颜色会显著加深，形成密度云效果
# 2. linewidth=0.5: 减小边框宽度，避免在点密集时白色边框掩盖了颜色
g.plot_joint(sns.scatterplot, s=100, alpha=0.3, edgecolor='white', linewidth=0.5)

# --- C. 绘制侧边分布图 (KDE) ---
g.plot_marginals(sns.kdeplot, fill=True, common_norm=False, alpha=0.3, linewidth=2)

# --- D. 标注两种模型的均值（标准差均值、接受率均值）---
# 需求：用更“优雅”的虚线标注，而不是用文字框。
# 做法：
# 1) 竖向虚线：Acceptance Rate 均值
# 2) 横向虚线：Std 均值
# 3) 文字用白色描边提升可读性（无 bbox 框）

# 保存当前坐标轴范围（用于把文字放在边缘附近）
xlim = g.ax_joint.get_xlim()
ylim = g.ax_joint.get_ylim()

dash_style = (0, (4, 4))  # 更轻盈的虚线

# 虚线风格：更细更淡一些
mean_line_width = 1.2
mean_line_alpha = 0.55

for _, row in means_df.iterrows():
    method = row['Method']
    x_mean = float(row['Acceptance_Rate'])
    y_mean = float(row['Std'])
    color = custom_palette.get(method, '#333333')

    # 均值虚线
    g.ax_joint.axvline(
        x_mean,
        linestyle=dash_style,
        color=color,
        linewidth=mean_line_width,
        alpha=mean_line_alpha,
        zorder=8,
    )
    g.ax_joint.axhline(
        y_mean,
        linestyle=dash_style,
        color=color,
        linewidth=mean_line_width,
        alpha=mean_line_alpha,
        zorder=8,
    )

    # 文字标注（不使用 bbox；用白色描边确保在点云上也清晰）
    # 竖线文字：贴在竖向均值线上，简要说明含义
    t1 = g.ax_joint.annotate(
        f"Mean Acc = {x_mean:.2f}%",
        xy=(x_mean, ylim[1]),
        xytext=(0, -6),
        textcoords='offset points',
        ha='center',
        va='top',
        fontsize=10,
        color=color,
        rotation=90,
        zorder=11,
        clip_on=True,
    )
    t1.set_path_effects([pe.withStroke(linewidth=3, foreground='white', alpha=0.9)])

    # 横线文字：贴在横向均值线上，简要说明含义
    t2 = g.ax_joint.annotate(
        f"Mean Std = {y_mean:.4g}",
        xy=(xlim[1], y_mean),
        xytext=(-6, 0),
        textcoords='offset points',
        ha='right',
        va='center',
        fontsize=10,
        color=color,
        zorder=11,
        clip_on=True,
    )
    t2.set_path_effects([pe.withStroke(linewidth=3, foreground='white', alpha=0.9)])

# ==========================================
# 5. 调整标签、标题与保存
# ==========================================
g.ax_joint.set_xlabel("MCMC Acceptance Rate (%)", fontweight='bold')
g.ax_joint.set_ylabel("Model Uncertainty (Standard Deviation)", fontweight='bold')

# 动态调整标题和图例
if ONLY_SHOW_UNIFORM:
    title_text = "Performance Diagnostics: Uniform Priors Only"
    # 如果只有一种数据，图例可能不需要，或者移动位置
    sns.move_legend(g.ax_joint, "upper right", bbox_to_anchor=(1, 1), frameon=True)
else:
    title_text = "Performance Diagnostics: Bayesian vs. Uniform Priors"
    sns.move_legend(g.ax_joint, "upper right", bbox_to_anchor=(1, 1), title=None, frameon=True)

plt.suptitle(title_text, y=1.02, fontsize=16, fontweight='bold', color='#333333')

plt.tight_layout()

# --- 保存图片 ---
# 根据模式生成不同的文件名
output_filename = 'diagnostics_uniform_only.png' if ONLY_SHOW_UNIFORM else 'diagnostics_comparison.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"图片已保存为: {output_filename}")

plt.show()