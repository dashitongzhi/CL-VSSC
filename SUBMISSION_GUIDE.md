# Paper 2 (`CL-VSSC`) 投递操作与期刊战略指南

> **现状总览**：除个人实名信息外，技术审稿与开源仓全套材料已完备。  
> - **公开代码仓**：[`https://github.com/dashitongzhi/CL-VSSC`](https://github.com/dashitongzhi/CL-VSSC)  
> - **9 页 `IEEEtran` 终稿**：[`CL-VSSC_ClosedLoop_Control_IEEE.pdf`](CL-VSSC_ClosedLoop_Control_IEEE.pdf)  
> - **英文投稿信**：[`submission/COVER_LETTER_IEEE.md`](submission/COVER_LETTER_IEEE.md)  
> - **投稿系统复制粘贴库**：[`submission/DECLARATIONS_AND_METADATA.md`](submission/DECLARATIONS_AND_METADATA.md)

---

## 一、投前 3 分钟个人填空清单（仅余这些 `TODO(PERSONAL_INFO)`）

在点击期刊系统 `Submit` 之前，全局搜索 `TODO(PERSONAL_INFO)`，只需替换以下 4 处个人信息：

1. **[`manuscript/main.tex`](manuscript/main.tex)**：
   - 第 `24` 行 `\markboth`：把 `[Author Surname]` 改为第一作者姓氏（若不投 *IEEE TCYB*，顺手改期刊英文名）。
   - 第 `31–33` 行 `\author` 与 `\thanks`：填写真实 `[Author Full Name]`、`[Department / School]`、`[University / Institution]`、`[City] [Postal Code], [Country]`、`[author@institution.edu]`、`[0000-0000-0000-0000]` 及基金号（无基金直接删掉基金那句 `\thanks`）。
   - 第 `274–278` 行（可选）：若目标刊要求初始投稿附带作者简介，取消注释 `IEEEbiography` 并填入个人简历与证件照。
2. **[`manuscript/references.bib`](manuscript/references.bib)**：
   - 第 `318–325` 行 `cao2026mivscl`：把 `Author, Corresponding` 替换为 Paper 1（`MIVS-CL`）的真实作者署名；若 Paper 1 已挂 arXiv，把 arXiv 编号填进 `note` 字段。
3. **[`submission/COVER_LETTER_IEEE.md`](submission/COVER_LETTER_IEEE.md)**：
   - 替换开头日期 `[TODO(PERSONAL_INFO): Date]` 与文末通讯作者署名、单位、地址、邮箱和 ORCID。
4. **[`submission/DECLARATIONS_AND_METADATA.md`](submission/DECLARATIONS_AND_METADATA.md)**：
   - 确认基金资助声明与 CRediT 作者贡献列表。

改完上述 4 处后，本地执行一次 PDF 重编译并推送更新即可直接上传投稿系统：
```sh
cd manuscript && tectonic --keep-logs main.tex && cp main.pdf ../CL-VSSC_ClosedLoop_Control_IEEE.pdf && cd ..
PYTHONPATH=/Users/kral/Documents/Research/MAVS-Reliability-2026-11-05/.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3.12 -B src/verify_paper.py
git add -A && git commit -m "chore: populate author and affiliation metadata" && git push origin main
```

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
