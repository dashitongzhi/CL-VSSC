# Paper 2 (`CL-VSSC`) 投递操作与期刊战略指南

> **现状总览**：除个人实名信息外，技术审稿、多期刊直接上传件、Graphical Abstract、Supplementary Material 与开源仓已**100% 完备**。  
> - **公开代码仓**：[`https://github.com/dashitongzhi/CL-VSSC`](https://github.com/dashitongzhi/CL-VSSC)  
> - **9 页 `IEEEtran` 主稿 & 补充材料 PDF**：[`CL-VSSC_ClosedLoop_Control_IEEE.pdf`](CL-VSSC_ClosedLoop_Control_IEEE.pdf) & [`CL-VSSC_Supplementary_IEEE.pdf`](CL-VSSC_Supplementary_IEEE.pdf)  
> - **可视化摘要（Graphical Abstract）**：[`submission/Graphical_Abstract.pdf`](submission/Graphical_Abstract.pdf) / [`PNG (300 DPI)`](submission/Graphical_Abstract.png)  
> - **多顶刊排版好的 PDF Cover Letter & 附件**：[`Cover_Letter_IEEE_TCYB.pdf`](submission/Cover_Letter_IEEE_TCYB.pdf) / [`Cover_Letter_IEEE_TNNLS.pdf`](submission/Cover_Letter_IEEE_TNNLS.pdf) / [`Cover_Letter_Information_Fusion.pdf`](submission/Cover_Letter_Information_Fusion.pdf) / [`Highlights_Elsevier.pdf`](submission/Highlights_Elsevier.pdf) / [`Declaration_of_Interest_Elsevier.pdf`](submission/Declaration_of_Interest_Elsevier.pdf)  
> - **一键填署名自动化脚本**：[`submission/author_profile.template.json`](submission/author_profile.template.json) + [`src/apply_author_info.py`](src/apply_author_info.py)

---

## 一、投前 1 分钟一键替换个人实名信息（推荐流程）

**方法 A（推荐，一条命令自动替换全部文件并重编全部 PDF）：**
1. 复制模板文件并填写真实个人信息：
   ```sh
   cp submission/author_profile.template.json submission/author_profile.json
   ```
   只需修改里面的真实姓名、院系/学校/邮编、邮箱、ORCID、基金说明与 Paper 1 真实作者署名（注意勿将个人隐私 `author_profile.json` 提交前核对）。
2. 运行一键写入 + 全套编译与一致性验收命令：
   ```sh
   PYTHONPATH=/Users/kral/Documents/Research/MAVS-Reliability-2026-11-05/.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3.12 -B src/apply_author_info.py --profile submission/author_profile.json --build
   git add -A && git commit -m "chore: populate author and affiliation metadata" && git push origin main
   ```

**方法 B（手工检索 `TODO(PERSONAL_INFO)`）：**
全局检索 `TODO(PERSONAL_INFO)` 亦可定位所有占位，对应文件包括 [`manuscript/main.tex`](manuscript/main.tex)、[`manuscript/supplementary.tex`](manuscript/supplementary.tex)、[`manuscript/references.bib`](manuscript/references.bib)、[`CITATION.cff`](CITATION.cff) 与 [`submission/`](submission/) 内的 Cover Letter 源码。

改完上述信息后，[`submission/IEEE_LaTeX_Source_Submission.zip`](submission/) 会由 `src/build_submission_bundle.py` 自动刷新（含 `main.tex`、`supplementary.tex`、`references.bib`、全部图/表，直接传 ScholarOne TeX Source 包即可）。

---

## 二、分层期刊投递策略与工作区投稿顺序

| 优先级 | 目标期刊 | 定位与适配度 | 模板 / 页数限制 | 初审与送审要点 |
|---|---|---|---|---|
| **首选主冲 (1st)** | ***IEEE Transactions on Cybernetics* (TCYB)** | 控制论 + 多智能体/网络化谱聚类最对口顶刊，长期青睐“闭环反馈 + 严格收敛/ISS 界 + 大规模图聚类” | `IEEEtran` 双栏；常规论文一般 10–12 页免超页费（当前 **9 页**，余量充裕） | 重点把 Eq. (7) ISS + Davis–Kahan 界、`Prop. 1` 零近似二分图等价、以及 `536` 点 $(K_p,K_i,\delta_{\mathrm{dz}})$ 相图放在 Cover Letter 首推亮点 |
| **同档主冲 (2nd)** | ***IEEE Transactions on Neural Networks and Learning Systems* (TNNLS)** | 覆盖神经网络与流形/图表示学习，对比基线含 `Attention-MOAE` / `Subtype-GAN` 等深度多模态模型 | `IEEEtran` 双栏；通常 10–12 页（当前 **9 页**可直接送） | 强调 `CL-VSSC` 为何能以正交闭环避免深度注意力的坏通道锁定（`0.866` vs `0.002`）与 `300k` 线性内存扩展 |
| **稳妥备选 (3rd)** | ***IEEE TSMC: Systems*** | 偏重系统级容错与多传感器信息融合，极其看重传感器故障阶跃响应与自愈时间 | `IEEEtran` 双栏；无模板改造成本 | 突出 `1,560` 次三模态传感器压测、BIBO 抗饱和以及 `N=200k` 7 步阶跃故障自愈 |
| **跨社备选 (4th)** | ***Information Fusion* (Elsevier)** | 多源异构信息融合顶刊 | 需由 `IEEEtran` 切换为 `elsarticle` 模板，并附上 `SUBMISSION_GUIDE` 与 `DECLARATIONS_AND_METADATA.md` 第 5 节的 5 条 `Highlights`（每条 `<=85` 字符） | 重点突出多模态可信融合与主模态可观性上限分析 |

### 与工作区其余论文的投稿区隔与节奏建议
1. **Paper 1 (`MIVS-CL`) vs Paper 2 (`CL-VSSC`)**：绝对不要隐瞒伴随关系。强烈建议**先将 Paper 1 投向 *IEEE JBHI* / *IEEE/ACM TCBB*（或先挂 arXiv 拿公开编号），再或同时向 *IEEE TCYB* 投 Paper 2**；投 Paper 2 时务必保留正文 `[29]` 的显式引用，并在 ScholarOne 的 *Supplemental Material for Review Only* 上传 Paper 1 PDF（已在 `COVER_LETTER_IEEE.md` 写好标准 IEEE PSPB 披露段落）。
2. **与 Paper 3 (`KGC-MPC`) / Paper 4 (`SPIC-MAS`) / Paper 5 (`GRAM-UIO`) 的区隔**：Paper 2 聚焦“无监督多模态聚类的闭环加权与谱间隙调控”，不涉及 Koopman 轨迹控制（P3）、空间 L–R 牵制（P4）或卡尔曼/UIO 状态观测（P5），天然错开审稿池与研究主题。

---

## 三、高频审稿质疑预判与直接回复口径

1. **Q：为什么不在 Paper 2 里一起跑缺失块插补对比？与伴随论文 `MIVS-CL` 增量边界在哪？**  
   - *口径*：`MIVS-CL` 处理临床块状缺测的掩码潜空间插补与生信预后解释；Paper 2 完全聚焦实时闭环控谱聚类与线性大规模扩展（Null-SNR 传感器、Double-Centered Anchor-Local PI、Prop. 1–3 理论界、`1,560` 故障压测、`536` 点相图与 `300k/200k` 扩展），二者图表 0 重叠、10-gram 重合率 `< 0.6%`。
2. **Q：你们在 `BMMC` 干净单细胞上的 NMI（`0.588–0.593`）略低于 `Attention-MOAE`（`0.633`）和 `Seurat-WNN`（`0.618`），如何自圆其说？**  
   - *口径*：这是正交线性 Stiefel 约束带来的“标称自由度 vs. 鲁棒可控性”帕累托权衡（已在第 VI 节直面讨论）。无约束注意力在干净数据高 `0.04`，但在 ADT $\sigma=3.0$ 故障下把权重锁死在 `0.866` 导致 ARI 暴跌 `-78.9%` 到 `0.051`，而 `CL-VSSC-Anchor` 保持 `0.169` 第一名并有严格 ISS 界。
3. **Q：为什么 DNA 甲基化重故障下 `Null-SNR PI` 保留 ARI 只有 `0.170`？**  
   - *口径*：这是物理**可观性极限（Observability Ceiling）**而非控制器失灵——控制器已成功把坏掉的甲基化权重压到 `0.003`（36/36 全胜，$q=1.46\times 10^{-11}$），但 GBM/KIRC 的真实分子分型主要由甲基化唯一决定，其余视图本身不含足够区分度（而在次级视图 mRNA/miRNA 故障上，Null-SNR PI 以 `0.796` ARI 全胜 9/9 基线，$q\le 6.19\times 10^{-4}$）。
4. **Q：为什么 `GSE194122-Expanded`（`150k–300k`）采用了扩展抽样？**  
   - *口径*：已如实标注 `90,261` 为实测真实全量细胞，超出 `90.3k` 的部分在正文和 `Fig. 6(c)` 用虚线明确标为 `Expanded 150k–300k`（仅用于验证硬件 $\mathcal{O}(N^{0.996})$ 显存线性扩展与 $N=200\text{k}$ 中途阶跃故障自愈收敛速度，不把扩增样本当作生物学准确度结论）。
