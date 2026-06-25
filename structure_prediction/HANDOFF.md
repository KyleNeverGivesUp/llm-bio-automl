# HANDOFF — PXR Structure (Pose) Prediction

> 本目录 `structure_prediction/` 承载 OpenADMET PXR challenge **structure / pose 赛道**的全部工作。
> 它是 `llm-bio-automl` 仓库下的一个**独立子目录**(不是单独 repo),与 activity track 代码隔离，但同仓库可直接复用编排 plumbing。
> **先读 Section 3「核心认知」再动手。**

## 0. 一句话

给定 184 个配体的 SMILES + PXR 靶点，**预测每个配体与 PXR 形成的蛋白-配体复合物 3D 结构**;用预训练结构模型**推理**(不训练)，多工具生成位姿 → 精修 → 合法性过滤 → 共识选优 → 打包成 `structures.zip` 提交。**主评分指标 = lDDT-PLI(越高越好)。**

## 1. 与 activity track 的关系 / 隔离

- 同仓库 `llm-bio-automl`，但所有 structure 相关代码、数据、产物**只放在 `structure_prediction/` 下**，不与 `src/`(activity)混。
- **能复用的只有薄 plumbing**:LLM 配置层、agent 编排骨架、断点续跑(见 Section 12)。其余 100% 新写。
- ⚠️ **依赖环境隔离**:结构工具(Boltz/DiffDock/OpenMM/OpenStructure…)是重依赖，会与 activity 的干净 sklearn 环境冲突。**给本目录单独建 venv / 单独 requirements**;重型工具实际跑在学校集群的 container(Apptainer)/conda 里(见 Section 9)，本地只放编排 glue。
- 同一套设计哲学:**content over orchestration**——80% 精力在内容(工具选型/前处理/共识)，20% 在 agent 编排。

## 2. 已核实的 challenge 事实

- 数据集:`openadmet/pxr-challenge-train-test`(HF)。Challenge Space:`huggingface.co/spaces/openadmet/pxr-challenge`。
- 结构赛道文件:`pxr-challenge_structure_TEST_BLINDED.csv`，config=`structure`，split=`test`，**184 行**。
- 列:`structure`(晶体 ID，如 `x00011-1`，XChem/Fragalysis 命名)、`smiles`、`Molecule Name`、`OCNT_ID`。
- **是 BLINDED(盲测)**:只给 SMILES + ID，**不给坐标**。整个 dataset repo 只有 CSV，无任何 .pdb/.cif/.sdf。
- 这 184 个分子**确实有真实 X 射线晶体结构**(组织方手里的答案，server 端用来评分)。
- 时间:Challenge 期 **2026-04-01 → 07-01**;Phase 2 全程盲测，**2026-07-01 截止**。

## 2.5 提交格式 & 评分(已核实，来自 Space 源码 app.py / models.py / submission_store.py / structure_leaderboard.csv)

- **提交物 = 单个 `structures.zip`**(canonical 文件名固定)，含 184 个预测的**蛋白-配体复合物结构**;经 Space 上传，存 `submissions/structure/{user}/{id}/structures.zip`。
- **节流:每 4 小时 1 次提交**(`HOURS_BETWEEN_SUBMISSIONS=4`)。**Coverage 是评分项 → 必须交满全部 184**(top 选手 1.00)。
- **评分 = OpenStructure / CASP 配体指标**(server 端用盲测晶体算):
  - **★ 主排名指标 = lDDT-PLI(↑，越高越好)** —— app.py 明确 `sort_values("LDDT-PLI_mean", ascending=False)`。
  - 次要(展示用):**BiSyRMSD(↓)**、**lDDT-LP(↑)**、**Ligand RMSD(↓)**、**Coverage(↑)**。
- **样例榜揭示方法天梯(强信号):** `alphafold_fan` 0.88 (BiSyRMSD 1.24) > `diffdock_enjoyer` 0.81 (1.67) > `rosetta_dock` 0.74 (2.13) > `gnina_user` 0.68 (2.58) > `autodock_42` 0.39 (4.31)。→ **co-folding 碾压经典 docking。**
- 存在 `model_report_link` + `used_proprietary_data` 字段 → 同样需**方法报告** + 专有数据披露。

## 3. 核心认知(必须先内化)

1. **不训练任何模型。** 结构预测主模型(Boltz/Chai/AF3)用预训练权重直接推理;docking(Vina/Gnina)本就是物理搜索不学习;数据太少，fine-tune 只会过拟合 → 不做。
2. **难点不在建模，在工程**:重型 GPU 推理、co-folding 的 MSA、前处理(口袋/质子化/互变异构)、精修、**无 GT 下的位姿选择**。
3. **LLM/agent 的角色 = 编排 + 领域判断 + 自修复，不是优化器。**
4. **co-folding 是主线，不是 docking。** 样例榜 lDDT-PLI:alphafold 0.88 ≫ diffdock 0.81 ≫ rosetta 0.74 ≫ gnina 0.68 ≫ autodock 0.39。**优先 Boltz-2 / Chai-1 / AF3**;docking 仅用于共识/多样性。

## 4. Ground truth 与优化信号(关键)

**两层，绝不能混:**

| 用途 | 有没有 GT | 信号 |
|---|---|---|
| **提交那 184 个** | 无(盲) | 只能用 `trust_score`(无监督，见 Section 7)做位姿选择 |
| **开发/调 pipeline** | **有(自己拼)** | **78 个公开 PXR 共晶 → 用 OpenStructure 算真·官方指标(lDDT-PLI / BiSyRMSD)** |

→ 核心策略:**用公开晶体在本地复刻官方 lDDT-PLI/BiSyRMSD 来选工具/前处理/共识权重**(这一步让 agentic 优化闭环真正成立)，再把调好的 pipeline 用到 184 盲测上。
→ 照搬 activity track 的教训:challenge 答案是盲的 → 自己造一个可量化、且**与官方同指标**的验证集。

## 5. 整体流程

```
①  受体准备     PXR LBD 结构(docking用)或仅序列(co-fold用);加氢/去水/定口袋
②  配体准备     SMILES → 清洗 → 质子化/互变异构态 → 3D 构象(RDKit ETKDG)
③  位姿生成     【推理，不训练】 主线 co-fold: Boltz-2/Chai-1/AF3 ; 辅 docking: Gnina/DiffDock
                 每个工具吐"多个候选复合物 + 自带置信度"
④  精修         OpenMM/RDKit 能量极小化，消冲突
⑤  合法性过滤   PoseBusters 删物理不合理位姿(硬门)
⑥  选择/共识    跨工具按对称校正 RMSD 聚类 → 共识簇 → trust_score 选终选复合物
⑦  提交         184 个复合物打包成 structures.zip (须确认内部命名/格式) 上传 Space

外层校准循环(决定①~⑥的配置):
78 个公开 PXR 共晶 → 跑候选 pipeline → OpenStructure 算 lDDT-PLI/BiSyRMSD → 选最优配置 → 用到 184 盲测
```

## 6. 系统架构:agent 角色 + IO 契约

| Agent | 输入 | 动作 | 产出 |
|---|---|---|---|
| **Manager** | task.json | 阶段编排、GPU/队列预算路由(便宜工具先筛→难的上 co-fold)、断点续跑 | `run_state.json` |
| **Setup** | 靶点 + SMILES 列表 | 受体 prep(取/备 PXR LBD、加氢、定口袋)、配体 prep(清洗、质子化、3D 构象) | `prepared_receptor.cif`、`ligands_prepared.sdf`、`pocket_def.json` |
| **Retrieval** | task_spec | (LLM)检索并排序工具 + 拉 PXR 先验(已知共晶 PDB、关键结合残基) | `tool_registry.json`、`domain_priors.json` |
| **Designer** | tool_registry + priors | (LLM)每个配体生成 N 个**多样化**配方=(工具,口袋模式,质子化态,采样数,seed) | `recipes.json` |
| **Coder** | recipes + 备好输入 | 每个配方→可跑 job(SLURM/本地)→跑工具→解析→OpenMM 极小化;**失败读日志自修复** | `poses/<lig>/<recipe>/complex.cif` + raw confidence |
| **Validator** | 所有位姿 | PoseBusters、strain energy、口袋内/外、关键残基接触 | `validity_report.json` |
| **Aggregator** | 位姿+validity+confidence | 按对称校正 RMSD(spyrmsd)聚类 → 共识簇 → trust_score 选终选 + 置信估计 | `final_complexes/` |
| **Exporter** | 终选复合物 | 按 Space 要求打包 structures.zip + 写自评可信度报告 | `submission/structures.zip` |

> 相对 activity 项目两处关键改名:**Selector→Validator**(无 RMSD 可选时按合法性筛);**Aggregator=consensus docking**(无监督)。但在**校准循环里 Selector 复活**——用 78 个公开晶体的真 lDDT-PLI 选配置。

## 7. trust_score(盲测 184 上的无 GT 评判核心)

```
先硬过滤:PoseBusters 不合法 / 配体在口袋外 → 淘汰
再加权(各项归一化 0–1):
trust = w1·合法性(PoseBusters+strain) + w2·跨方法共识 + w3·模型置信 + w4·先验命中
起步权重 w = [0.30, 0.35, 0.20, 0.15]   # 共识权重最高:唯一跨方法的客观信号
```

- **共识 w2 ⭐**:看有几个**独立工具/seed** 收敛到同一簇(不是同一工具采样多)。
- **权重用 78 个公开晶体校准**——目标是让 trust_score 与真·lDDT-PLI 尽量相关(别拍脑袋)。
- co-folding 的自带 confidence(Boltz confidence / ipTM / PAE)常和 lDDT-PLI 相关，是 w3 的主来源。

## 8. 工具栈

| 角色 | 工具 |
|---|---|
| **主力 co-folding(主线)** | **Boltz-2** / Chai-1 / AF3(盲预测最稳，lDDT-PLI 最高，自带 confidence) |
| docking 做共识 | Gnina(CNN 打分)、DiffDock-L、Uni-Mol Docking v2、AutoDock Vina |
| 合法性 | **PoseBusters**(硬门) |
| 精修 | OpenMM、RDKit MMFF |
| 前处理 | P2Rank/fpocket(口袋)、Dimorphite-DL(质子化)、Meeko(受体 prep) |
| **本地评分(复刻官方)** | **OpenStructure `ost compare-ligand-structures`**(lDDT-PLI / BiSyRMSD / lDDT-LP) |
| 对称校正 RMSD(聚类用) | spyrmsd |
| 验证基准 | **78 个公开 PXR 共晶(RCSB)**、PoseBusters benchmark、Astex Diverse、PDBbind core |

## 9. 算力(L40S / SLURM)

- 计划用**学校集群的 L40S(48GB)**——对 PXR(LBD ~280–350 残基 + 小分子)绰绰有余，Section 8 全部工具都能本地跑。
- 集群坑:**SLURM 排队 + wall-clock 上限** → Manager 要切分作业 + checkpoint;**多半需 Apptainer/conda 且可能无外网** → 提前把模型权重下载缓存到 scratch;注意 scratch 配额。

## 10. 目录骨架(在 `structure_prediction/` 下)

```
structure_prediction/
  HANDOFF.md                  ← 本文件
  pyproject.toml / requirements.txt   # 独立依赖，勿混 activity 环境
  src/
    agent/      manager / setup / retrieval / designer / coder / validator / aggregator / exporter
    tools/      boltz_runner.py · gnina_runner.py · diffdock_runner.py     # 全新
    scoring/    pose_validity.py(PoseBusters) · ost_score.py(OpenStructure) · consensus.py · trust_score.py
    prep/       receptor_prep.py · ligand_prep.py
  benchmarks/   78 个公开 PXR 共晶 + PoseBusters
  data/         structure_TEST_BLINDED.csv
  outputs/<run_id>/
```

## 11. 落地 roadmap

| 阶段 | 内容 | 产出 |
|---|---|---|
| **P0 MVP** | Boltz-2 单工具:Setup→Coder 跑→top-1 by confidence→PoseBusters 过滤→打包 structures.zip | 已能交一版 |
| **P0.5 本地评分** | 接 OpenStructure，在 1~2 个公开 PXR 共晶上跑通 lDDT-PLI/BiSyRMSD 计算 | 有了真·指标 |
| **P1 共识** | 接 Gnina/DiffDock + consensus + trust_score | 质量主升 |
| **P1.5 校准** | 在 78 个公开 PXR 共晶上用官方指标调工具/权重 | **优化闭环成立** |
| **P2 先验** | Retrieval 拉关键残基，w4 上线 | 更有依据 |
| **P3 自修复** | Coder 读 SLURM 报错自动修 | 鲁棒性 |
| **P4 Designer 多样化** | LLM 真正生成多样配方 + Manager 预算路由 | 自动化 |

**先 P0 跑通再逐步加**——P0 几乎没 agent 价值就出来了;agent 层最后加。

## 12. 可复用 plumbing(从 `llm-bio-automl` 的 `src/agent/`)

- `LLM_base.py`(LLM/OpenRouter 配置层)
- `agent_context.py`(RunContext/RunState/AgentResult)
- Manager 的编排 + run_state 落盘 + resume 骨架

同仓库可直接 `import`，但**建议复制进 `structure_prediction/src/agent/`** 以免被 activity 侧改动牵连。其余(tools/scoring/prep)100% 新写。

## 13. Open questions 状态

| # | 问题 | 状态 |
|---|---|---|
| 1 | 提交格式 | ✅ **解决**:单个 `structures.zip`，184 个蛋白-配体复合物;4h 节流;须交满 184 |
| 1b | **zip 内部布局/命名**(每个预测文件如何对应 184 个 target、用 `structure` ID 还是 SMILES、单文件格式 .cif/.pdb) | ⏳ **唯一剩余阻塞** —— 源码只锁定了外层 zip;内部约定要从 challenge 文档/blog/样例确认 |
| 2 | PXR 公开共晶 | ✅ **解决**:RCSB 有 **78 个**带配体的人源 PXR 结构(1ILH/1NRL/2O9I/1M13/1XV9/2QNV…) |
| 3 | 评分指标 | ✅ **解决**:主 = lDDT-PLI(↑);次 = BiSyRMSD↓/lDDT-LP↑/Ligand RMSD↓/Coverage↑;OpenStructure 算 |
| 4 | co-fold vs dock | ✅ **解决**:样例榜证明 **co-folding 主线**(alphafold 0.88 ≫ docking) |
| 5 | 截止/方法报告 | ✅ **解决**:2026-07-01 截止;需 model report + 专有数据披露 |
| 6 | structure phase-1 unblinded | ❎ 数据集无该 config → 没有官方放出的结构答案;靠 78 个公开共晶 |

## 14. 诚实边界

- 对那 184 个**你永远不知道实际多准**，只能报 confidence;trust_score 高 ≠ 正确。
- agent **不提升准度**，只提升流程鲁棒性、产出合法性、选择依据。
- 准度上限由 **foundation model 强弱(co-folding)+ 前处理质量**决定，不是 agent 层。
- **完全没有训练循环**;唯一的"调"是 pipeline 配置选择(用 78 个公开晶体的官方指标)。
