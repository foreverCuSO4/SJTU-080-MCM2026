### 1. 约束满足与拒绝采样 (Constraint-Based Rejection Sampling)
这是模型的基础骨架。
*   **方法描述：** 你没有使用“黑箱预测”，而是构建了一个基于历史事实的**逆向求解器**。你将“谁被淘汰”作为**硬约束（Hard Constraint）**，通过蒙特卡罗模拟生成成千上万种可能的粉丝投票分布，并利用**拒绝采样算法**剔除所有与历史事实（Ground Truth）相悖的解。
*   **价值：** 保证了模型输出的**“反事实有效性”**。每一个保留下来的样本，在逻辑上都是完全可能导致真实历史发生的。

### 2. 贝叶斯重要性采样 (Bayesian Importance Sampling)
这是你为了解决“不确定性”引入的核心优化。
*   **方法描述：** 你抛弃了简单的“独立同分布（i.i.d.）”假设，引入了**马尔可夫性质（Markovian Property）**，即本周的粉丝排名与上一周高度相关。
*   **数学亮点：** 你使用了**高斯核函数（Gaussian Kernel）**构建概率权重：
    $$ w_i \propto \exp\left(-\frac{\text{Distance}^2}{2\sigma^2}\right) $$
    这在统计学上被称为**重要性采样**。它不是简单粗暴地删掉“波动大”的数据，而是根据其发生的**似然概率（Likelihood）**赋予权重。
*   **价值：** 这将模型从一个简单的随机生成器提升为一个**状态估计器（State Estimator）**，能够量化“粉丝忠诚度”对排名的影响。

### 3. 幸存者重排归一化 (Survivor Re-ranking Normalization)
这是最体现你“洞察力”的细节修正（对应代码中的 `get_adjusted_expectations`）。
*   **方法描述：** 你指出了朴素时间模型中的**“结构性偏差”**。当高排名选手淘汰时，剩余选手的排名自然上升（例如第2名变第1名）。这属于**结构性位移**，而非粉丝情绪波动。
*   **解决方案：** 你引入了**条件期望排名（Conditional Expected Rank）**。在计算波动时，你比较的是“当前排名”与“基于上周表现且剔除淘汰者后的期望排名”。
*   **价值：** 这是一个非常细腻的**偏差修正（Bias Correction）**。它证明了你的模型能够区分**“系统性变化”**（比赛机制导致的）和**“随机性变化”**（粉丝意愿导致的），极大地提高了模型的可信度。

### 4. 加权不确定性量化 (Weighted Uncertainty Quantification)
这是对题目要求“Provide measures of certainty”的直接回应。
*   **方法描述：** 你没有使用普通的标准差，而是计算了**加权标准差（Weighted Standard Deviation）**。
    $$ \sigma_{weighted} = \sqrt{\frac{\sum w_i (x_i - \mu)^2}{\sum w_i}} $$
    并且，你引入了**有效样本量（Effective Sample Size, ESS）**的概念：
    $$ ESS = \frac{(\sum w_i)^2}{\sum w_i^2} $$
*   **价值：** 这为你的估计结果提供了一个严谨的统计学置信度边界。ESS 指标还能让你评估在引入了时序约束后，我们到底“保留”了多少有效信息量。

---

### 论文写作话术建议 (Copy-Paste Ready)

为了方便你写论文，这里有一段针对上述方法的英文摘要草稿：

> **3.2 Inverse Inference Model based on Bayesian Importance Sampling**
>
> To estimate the unobservable fan votes, we developed a Monte Carlo simulation framework grounded in **Constraint Satisfaction**. We treated the actual elimination results as "Hard Constraints" to filter feasible fan ranking permutations via **Rejection Sampling**.
>
> However, simple rejection sampling assumes temporal independence, ignoring the inertia of fan bases. To address this, we incorporated a **Bayesian Importance Sampling** mechanism. We modeled the week-to-week rank transitions using a **Gaussian Kernel**, assigning higher probability weights to ranking trajectories that exhibit stability consistent with fan loyalty ($\sigma$).
>
> **3.3 Correction for Structural Bias: Survivor Re-ranking**
>
> A naive temporal comparison suffers from **"Survivor Bias"**. When a contestant is eliminated, the ranks of all surviving contestants naturally shift upward. This structural shift is distinct from fan sentiment volatility. We introduced a **"Survivor Re-ranking Normalization"** step (Eq. 5), where the current rank is compared against the *Conditional Expected Rank* of the survivors, rather than their absolute rank from the previous week. This decoupling ensures our uncertainty metrics reflect genuine fan volatility, not mechanistic artifacts.