# Paper 2 (`CL-VSSC` & `CL-VSSC-Anchor`) 控制理论精讲与上交 816 复试通关手册

> **本目录定位**：纯粹独立的自动控制与大规模可扩展多模态谱聚类论文（**9 页** IEEE Transactions 长文，目标期刊：*IEEE TCYB* / *TNNLS* / *TSMC-S* / *Information Fusion*，以及**上海交通大学 816 自动控制原理研究生复试代表作**）。  
> **配套论文终稿**：[`CL-VSSC_ClosedLoop_Control_IEEE.pdf`](CL-VSSC_ClosedLoop_Control_IEEE.pdf)（TeX 源码 [`manuscript/main.tex`](manuscript/main.tex)，31 篇规范引用，含在线反馈优化 OFO 与张量化二分图最新顶刊对比；投稿指南与 Cover Letter 见 [`SUBMISSION_GUIDE.md`](SUBMISSION_GUIDE.md) 及 [`submission/`](submission/)，公开仓库 [`https://github.com/dashitongzhi/CL-VSSC`](https://github.com/dashitongzhi/CL-VSSC)）。

---

## 第一章 用经典控制框架拆解全篇架构（Fig. 1）

这篇论文把前人“开环多视图分解 + 图融合 + 谱聚类”改造为一个标准的**离散时间双闭环反馈控制系统**：

1. **被控对象（Orthogonal Plant，Eq. 1）**：演化在互相正交 Stiefel 流形上的共享表征 $Z(t) \in \mathbb{R}^{n \times r}$（满足 $Z^\top Z = I_r$）与通道特异扰动因子 $U^{(v)}(t) \in \mathbb{R}^{n \times r_v}$（满足 $Z^\top U^{(v)} = 0$）。预计算各路传感器矩阵的平方谱范数 $s_v^2 = \sigma_{\max}^2(X^{(v)}) + 10^{-8}$ 进行尺度均衡定标，输出各路相似图 $A^{(v)}$ 或细胞–锚点二分图 $\tilde{B}^{(v)}$。
2. **内环执行器——对偶积分特征间隙控制器（Dual Eigengap Actuator，Eq. 2）**：
   - 被调量：当前聚类违反 Ky Fan 完美 $c$ 连通分量定理的迹偏差 $J_{\mathrm{spec}}(t) = \operatorname{Tr}(F^\top L(S) F)$；
   - 控制律：设定安全阈值 $\epsilon_{\mathrm{spec}}=0.02$，经由预热延时（$t \ge 15$）和上下限幅的离散对偶积分器更新 $\mu(t+1) = \operatorname{clip}(\mu(t) + \eta_\mu(J_{\mathrm{spec}}(t+1) - \epsilon_{\mathrm{spec}}),\; 0,\; \mu_{\max})$，再对跨簇虚假弱连边按谱距离平方 $H_{ij}=\|\hat{f}_i-\hat{f}_j\|_2^2$ 进行非负修剪 $\tilde{S}_{\mathrm{tgt}}$ 并做指数滑动平均（EMA）平滑执行 $S(t+1) = (1-\eta_S)S(t) + \eta_S \hat{S}_{\mathrm{tgt}}(t)$。
3. **外环传感器与 PI 单纯形控制器（Eq. 3 – Eq. 6）**：
   - **传感器（Eq. 4）**：$e_v(t) = \frac{1}{c_{\mathrm{norm},v}}\frac{1}{c} J_{\mathrm{spec},v} + \frac{\gamma_{\mathrm{cons}}}{c}\operatorname{Tr}(F^\top L_v F) + \gamma_{\mathrm{rec}} \tilde{r}_v(t)$。当 $c_{\mathrm{norm},v}=1$ 时即 **Smooth PI**；当基于**列随机置换洗牌零基线**取 $c_{\mathrm{norm},v} = \max(0.05, c_v^{\mathrm{null}})$ 时即 **Null-SNR PI**（校准通道固有维度信噪比偏差）。
   - **带泄露阻尼积分器的 PI 镜像下降控制器（Eq. 5，核心控制律！）**：
     $$I_v(t+1) = (1-\beta_I) I_v(t) + \beta_I d_v(t), \quad w_v(t+1) = \frac{w_v(t)^{1-\eta_w}\exp\!\left(-\frac{\eta_w}{\tau}\bar{u}_v(t+1)\right)}{\sum_{j=1}^V w_j(t)^{1-\eta_w}\exp\!\left(-\frac{\eta_w}{\tau}\bar{u}_j(t+1)\right)}$$
     - $(1-\beta_I)$ 泄露因子（$\beta_I=0.40$）等价于**抗积分饱和（Anti-Windup）与 BIBO 一阶低通滤波**：满足 $0 \le I_v(t) \le \max_{0 \le s < t} d_v(s)$，避免初期基底未稳定时的瞬态尖峰把积分器锁死，同时稳态严格收敛到 $I_v^* = d_v^*$；
     - $\delta_{\mathrm{dz}} = 0$ 平滑连续控制（**Smooth PI**）消除了硬死区控制盲区。
   - **李雅普诺夫型内部能量监控（Eq. 6）**：$\mathcal{V}(t)$ 耦合加权重构残差、对偶谱惩罚、熵势阱与共识割代价；实测在 `9/10` 可分 TCGA 队列上严格单调下降，而在非分层队列 `OV` 保持平直（`+0.004`），起到实时无监督物理监测表的作用。

---

## 第二章 正文 7 个公式与附录 A 三大命题（Prop. 1 – Prop. 3）精讲

### 2.1 为什么 90,261 单细胞不会爆显存？＋ 实战排查“局部积分漂移” Bug（Eq. 3 & Prop. 1）

1. **Proposition 1（Anchor Gram 严格谱等价定理——无近似地把 $\mathcal{O}(n^3)$ 压成 $\mathcal{O}(nm^2+m^3)$）**：
   - 在初始状态 $Z(0)$ 上用 MiniBatch $k$-means 挑 $m \ll n$（如 `384`）个地标，建 $n \times m$ 行随机二分图并双随机归一化得到 $\tilde{B} = D_r^{-1/2} B D_c^{-1/2}$。
   - 由矩阵 SVD 左右奇异向量等价性，隐式 $n \times n$ 两跳拉普拉斯 $L_{\mathrm{bip}} = I_n - \tilde{B}\tilde{B}^\top$ 的前 $c$ 小特征向量 $F$ 与仅对 **$m \times m$ 小 Gram 矩阵 $\tilde{B}^\top \tilde{B}$** 做特征分解后左乘 $\tilde{B} U_{\mathcal{A},:,1:c}\Sigma_{1:c}^{-1}$ **数学上完全严格等价，零近似误差**！
   - 效果：10k 细胞 A100 耗时压缩 **`20.3×`**（`4.60 s` vs `93.58 s`）、显存压缩 **`13.3×`**（`603 MB` vs `8,002 MB`），`90,261` 全量样本仅 **`16.57 s / 5,295 MB`**。

2. **Double-Centered Anchor-Local PI（Eq. 3 `eq:p2_local`——非常加分的复试排雷故事！）**：
   $$\hat{F}_{\mathrm{anc}} = \operatorname{rownorm}(D_c^{-1} B_S^\top \hat{F}), \quad l_{jv} = \frac{\sum_i B_{ij}^{(v)} \|\hat{f}_i - \hat{f}_{\mathrm{anc},j}\|_2^2}{2\sum_i B_{ij}^{(v)} + \varepsilon}, \quad \tilde{\delta}_{jv}^{\mathrm{loc}} = \operatorname{clip}\!\left(l_{jv} - \bar{l}_{\cdot v} - \bar{l}_{j\cdot} + \bar{l}_{\cdot\cdot},\; -0.25,\; 0.25\right)$$
   - **工程踩坑过程**：单细胞中不同亚群分别依赖基因表达（2,000 维 RNA）或表面蛋白（134 维 ADT），需要为不同锚点学习局部权重 $W_{\mathrm{loc}}\in\mathbb{R}^{m\times V}$。起初若直接把未修正的 $l_{jv}$ 送进 PI 积分器，跨模态总体基线难度偏差会在迭代中不断被积分放大（Global Integral Drift），造成合成正交对比实验严重崩塌。改用“**减行均值 + 减列均值 + 加回总均值**”双重中心化之后，局部调控误差全局期望严格归零，在 L4 GPU 上仅多占 **`+2.6%` 显存**（`5,434 MB`），稳定提升 `BMMC-10k` ARI（`0.201 -> 0.218`, ACC `0.308 -> 0.316`）与 `BMMC-30k` ARI（`0.212 -> 0.222`）。

### 2.2 为什么一定要积分控制且不要死区？（Prop. 2）

- **Proposition 2**：Eq. (5) 是概率单纯形上带 KL 正则强凸优化的精确 KKT 闭式解，全程轨迹有界且严格留在单纯形内部 $\operatorname{int}(\Delta^V)$。在平滑 PI（$\delta_{\mathrm{dz}}=0$）平衡点，受损通道 $b$ 相对干净通道 $g$ 超出 $\Delta e = e_b^* - e_g^* > 0$ 时，权重比呈**精确指数衰减**：
  $$\frac{w_b^*}{w_g^*} = \exp\!\left(-\frac{1+\kappa_I}{\tau}\Delta e\right)$$
  反之加入硬死区 $\delta_{\mathrm{dz}}>0$（如 `0.14`）时衰减指数被削弱为 $\max(0,\Delta e - \delta_{\mathrm{dz}})$；一旦 $\Delta e \le \delta_{\mathrm{dz}}$，反馈彻底卡死（实测 `BMMC-10k` ADT 50% dropout 故障权重死卡在 `0.500`，ARI 跌到 `0.123`），而 `P-only`（$\kappa_I=0$）缺少积分积累，BIC mRNA 故障稳态残差高出近 3 倍（`0.031` vs `0.011`）。

### 2.3 双闭环协同稳定定理：ISS + Davis–Kahan $\sin\Theta$（Prop. 3 / Eq. 7）

$$\|\sin\Theta(F^*, F_\infty)\|_F \le \frac{2\sqrt{c}}{\Delta\lambda_c^* - \Delta_\infty} \left(\Delta_L \sum_{b \in \mathcal{B}} e^{-\frac{1+\kappa_I}{\tau}\Delta e_b} + \bar{\delta}_\mu\right)$$

- **为什么说“双环非联手不可”？**
  - **外环 PI 管分子**：把故障传感器注入的图拉普拉斯扰动 $\Delta_L$ 通过指数项 $e^{-\frac{1+\kappa_I}{\tau}\Delta e_b}$ 硬生生压缩到接近 `0`（如 `0.002` Null-SNR）；
  - **内环对偶控制 $\mu(t)$ 管分母**：EMA 状态更新保证对称压缩率 $(1-\eta_S)$，使系统满足 Sontag **输入-状态稳定性（ISS）**（收敛到终值误差球 $\Delta_\infty$），同时对偶修剪放大安全谱间隙 $\Delta\lambda_c^*$，扩大分母 $\Delta\lambda_c^* - \Delta_\infty$！两者配合才能从分子分母双向锁死特征子空间角度失真。

---

## 第三章 实验核心结论与诚实边界（Table I–III, Fig. 2–5）

1. **三模态 `1,560` 次故障压力测试（Table II & Fig. 3）**：
   - $\sigma=4.0$（60% dropout）下，Smooth PI 压制故障权重至 **`0.012`**（严于 `MCGC 0.190` 15.8 倍、`Seurat-WNN 0.264` 22.0 倍、`Attention-MOAE 0.319` 26.6 倍；`36/36` 全胜，单侧配对 Wilcoxon BH-FDR `q = 1.46×10^-11`）；加 `Null-SNR PI` 进一步压紧至 **`0.002`**，总体干净标签 ARI 提升 **`+0.103`**（`0.484 -> 0.587`，全部 10 组对比配置第一，BH-FDR 在 $\sigma=4.0$ 下 `q <= 0.0196` 胜 8/9 方法；在次级模态 mRNA+miRNA 故障 `n=24` 上均值 ARI 高达 **`0.796`**，以 `q <= 6.19×10^-4` 全胜全部 9 个基线）。
   - **诚实可观性极限**：在 DNA 甲基化重故障下 `Null-SNR` 保留 ARI (`0.170`) 略低于 `Smooth PI` (`0.214`)，因为在 GBM/KIRC 中甲基化本身是决定亚型的唯一主导模态——当主机传感器自身彻底报废时，即使控制权重压到 `0.003`，剩下来的从属视图在物理上也不足以恢复完整分型。
2. **`90,261` 单细胞 CITE-seq 与 ADT 故障反转（Table III & Fig. 4）**：
   - 干净数据集上 Seurat-WNN / Attention-MOAE NMI 略高（`0.618–0.633` vs `0.588–0.593`），因为本法约束在正交线性隐空间以换取可解释性与双环可控；而 `Anchor (Local+Null)` 在 `9/12` 多站点配对测试中干净 ARI 显著高于全局 `CL-VSSC-Anchor`（`p = 0.032`）。当 `BMMC-10k` 表面蛋白 ADT 遭 $\sigma=3.0$、50% 丢包冲击时，`Attention-MOAE` 的静态注意力把权重死锁在坏通道（`w_ADT = 0.866`），ARI 暴跌 **`-78.9%`**（`0.242 -> 0.051`），而 **`CL-VSSC-Anchor` 排名第一（`ARI 0.169 / ACC 0.282`）**（在 $\sigma\in\{1.5,3.0\}$ 6 组配对测试中对 `Attention-MOAE`/`Subtype-GAN`/`DZ-PI`/`w/o FB` `6/6` 全胜，`p = 0.0156`）。
3. **三维敏感性帕累托前沿（Fig. 5）**：
   - $(\kappa_I \times \tau)$ 网格证实提高 $\kappa_I$ / 降低 $\tau$ 可把 $\sigma=4.0$ mRNA 故障 ARI 拉到 BIC **`0.597`** / KIRC **`0.990`**，但过硬参数（$\tau=0.10, \kappa_I=0.35$）会削弱干净生存（BIC `0.719`），验证默认 $(\kappa_I=0.35, \tau=0.20)$ 处于稳健与标称最优盆地；$\mu_{\max}\in[0, 0.24]$ 推动 BIC 谱间隙单调翻倍（`0.0351 -> 0.0680`）；$m \in [128, 768]$ 在 L4 全程 `4.02–6.62 s` 内单调提升精度。

---

## 第四章 上交 816 自动控制复试高频问答 8 题（完整答辩口径）

1. **Q1：用自控标准框图术语介绍你的系统：Plant、Sensor、Actuator、Controller 分别是什么？**  
   - 被控对象（Plant）是 Stiefel 正交潜空间演化模块；传感器（Sensor）是观测割违反率 + 共识一致率 + 自由度校验残差 $e_v(t)$；执行器（Actuator）是单纯形加权融合及跨簇边修剪单元；双闭环控制器分别为外环带泄露积分的单纯形 PI 镜像下降控制器与内环对偶 Ky Fan 积分器 $\mu(t)$。
2. **Q2：为什么纯比例（P-only）会有稳态静差？为什么积分器还要加泄露因子 $(1-\beta_I)$？为什么去除死区（DZ-PI）？**  
   - 对持久阶跃型传感器故障，无记忆比例控制必须靠残留正误差维持抑制梯度，必然留下稳态静差（`BIC mRNA 0.031` vs PI `0.011`）。$(1-\beta_I)$ 起到抗积分饱和和有界稳态保障；硬死区（`0.14`）会在中等 SNR 差异（如 CITE-seq ADT 50% dropout）时令误差落在死区内部、使控制器彻底闭目锁死在 `0.500`。
3. **Q3：如何向控制老师解释 Proposition 3（ISS + Davis–Kahan $\sin\Theta$ 界）的物理意义？**  
   - EMA 状态融合的 $(1-\eta_S)$ 收缩率保证输入干扰有界时误差收敛到有限终值球 $\Delta_\infty$（满足 Sontag ISS 定义）；结合 Davis–Kahan 子空间扰动定理，最终分型角度偏差取决于“故障通道残余加权干扰（分子）÷ 有效谱间隙（分母）”，所以外环 PI 用指数率压分子、内环 $\mu$ 扩分母，实现协同稳定。
4. **Q4：Lyapunov 能量 $\mathcal{V}(t)$ 为什么在 10 个队列里只有 `OV` 是平的（`+0.004`）？**  
   - 所有 8 种主方法在 `OV` 队列上都未能找到显著生存划分（缺乏客观分子亚结构）。`V(t)` 没有靠虚假过拟合把能量强行压下去，说明它真实反映了闭环物理收敛状态，可做无监督“系统可聚类性实时诊断器”。
5. **Q5：`Null-SNR PI` 虽然让重故障 ARI 提升 `+0.103`，却使干净 TCGA 显著从 `5/10` 降为 `3/10`，算翻车吗？**  
   - 不算翻车，这正是控制工程经典的 **Nominal Performance vs. Disturbance Rejection Pareto Trade-off**。通过列洗牌基准定标，系统会对低维但对部分干净样本有补充价值的 miRNA 权重压紧（`0.035–0.099`），从而把极端故障泄露再砍近 7 倍至 `0.002`。Paper 2 里 `Fig. 5(a)` 明确给出了按场景选型的量化准则。
6. **Q6：9 万细胞怎么避免 $\mathcal{O}(n^3)$ 特征分解？有没有丢失精确度？**  
   - Proposition 1 严格证明了 $n\times m$ 双随机二分图 $\tilde{B}$ 的两跳拉普拉斯 $I_n - \tilde{B}\tilde{B}^\top$ 与 $m\times m$ 小 Gram 矩阵 $\tilde{B}^\top\tilde{B}$ 非零特征值完全等价、特征向量一 一线性投影对应，无任何随机近似误差。
7. **Q7：在做局部锚点 PI 时踩过什么真实科研坑，怎么解决的？**  
   - 局部切图损失直接积分会累积跨模态直流基准偏差（Global Integral Drift）。我们通过按锚点与按模态的双重中心化残差（ANOVA 式交互提取）并限幅 $\pm 0.25$ 叠加到全局误差，彻底消除全局漂移，只增 `2.6%` 显存就稳定拉升 `BMMC-10k/30k` 准确率（先由本地合成测试 `smoke_test.py` 定位验证后再上云实跑）。
8. **Q8：论文参数是按不同数据集调出来的吗？未来怎么进一步扩展？**  
   - 全局超参数一套固定到底（$r=24, r_v=6, \kappa_I=0.35, \tau=0.20, \beta_I=0.40, \mu_{\max}=0.12$），从不按单数据集手动特调；Stiefel 极收缩、单纯形与双限幅从结构上杜绝梯度爆炸。未来工作聚焦两项：利用 $\Delta\lambda_c(t)$ 曲线在线自适应辨识聚类数 $c$，并在特定子空间 $U^{(v)}$ 中加入显式多中心批次正交投影。

9. **Q9（v4 A100 80GB 核心追问 — 对接考研 816 经典控制稳定裕度与 OFO 视角）：$(K_p, K_i, \delta_{\mathrm{dz}})$ 相图与 $200\text{k}\text{--}300\text{k}$ 中途阶跃故障自愈如何体现经典控制理论？**
   - **理论对接（Online Feedback Optimization, Hauswirth et al., *Annual Reviews in Control* 2024）**：把迭代算法视作与被控对象实时闭环的反馈控制器，有效环路增益 $K_{\mathrm{eff}} = K_p(1+\kappa_I) = \frac{\eta_w}{\tau}(1+\kappa_I)$ 越大，Prop. 3 界分子衰减越快，但增益过高会突破离散小增益收缩域引发振荡甚至单模态锁死。
   - **稳定裕度相图（Fig. 6a–b，`results_v4/pi_stability_phase_v4.csv` 536 点）**：将镜像下降步长等价映射为经典比例增益 $K_p = \eta_w/\tau \in [0.25, 40.0]$ 与积分比 $\kappa_I = K_i/K_p \in [0, 10]$。当 $K_p \le 1.0$ 且 $\kappa_I$ 过小时进入 **欠驱动区（Under-Actuated）**（$w_{\mathrm{mRNA}}>0.05$）；当 $K_p \ge 6\text{--}10$ 或 $K_i/K_p \ge 4\text{--}10$（出现 $\Delta\mathcal{V}>0$）时进入 **过增益坍缩/失稳区（Collapse / Unstable）**；二者之间是带非单调振荡的 **过渡振荡带（Oscillatory）**，而默认参数 $(K_p=1.5, \kappa_I=0.35)$ 稳居 **阻尼收敛稳定域（Convergent）** 内部（BIC $4$ 步、KIRC $5$ 步收敛）。死区 $\delta_{\mathrm{dz}} \in \{0, 0.05, 0.14, 0.25\}$ 增大时，98 点公共网格中的失稳点数从 `56 -> 31 -> 1 -> 0` 下降，但欠驱动停滞区从 `13 -> 21` 扩大，直观验证了经典控制中“稳定裕度 vs 稳态灵敏度”的折中边界。
   - **超大规模线性扩展与中途阶跃自愈（Fig. 6c–d，`scale_200k_v4.csv` & `fault_self_healing_200k_v4.csv`）**：在 `A100-SXM4-80GB` 上从 `GSE194122` 真实 `90,261` 细胞扩展至 `GSE194122-Expanded` `300,000` 细胞（$>100\text{k}$ 切换 `MiniBatchKMeans`），峰值显存呈 $\mathcal{O}(N^{0.9955})$ 严格线性增长（`594.8 MB -> 17,546.7 MB` Smooth PI；`300k` 耗时 `53.56 s` / Local PI `52.98 s`）。在 $N=200{,}000$ 规模下于 $t=10$ 注入中途阶跃 ADT 故障（$\sigma=3.0, 50\%$ 丢弃），`Smooth / Null-SNR / Local PI` 均在 **7 次迭代（`3.82 s / 5.20 s / 6.03 s`）内完成 90% 自愈**（`ARI` 分别为 `0.174 / 0.195 / 0.203`），而 `DZ-PI`（$\delta_{\mathrm{dz}}=0.14$）停滞在 `0.500`（`ARI 0.136`），复现了 Prop. 2 死区阻塞与 Prop. 3 小增益 ISS 结论。
