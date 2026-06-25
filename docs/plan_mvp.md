# PXR Activity AutoML MVP Plan

## 目标

基于 Hugging Face `openadmet/pxr-challenge`，先实现一个最小可运行的 AutoML 闭环，只覆盖 `Activity Prediction` 赛道。

MVP 只要求做到：

1. 读入 PXR activity 数据
2. 生成结构化任务定义
3. 自动构造少量基线方案
4. 运行训练与交叉验证
5. 选择最优方案
6. 导出符合提交格式的 `submission.csv`

## 实现策略

目标架构按 `7 agents` 设计，但实现顺序按 `1 个主 agent 的 MVP` 落地。

也就是说：

- 架构目标是：`Manager / Setup / Retrieval / Designer / Coder / Tuner / Exporter-Aggregator`
- 当前 MVP 运行方式是：先由一个主控 orchestrator 顺序调用各模块
- 当前代码先按未来 7 个 agent 的职责拆模块
- 等单主控链路稳定后，再把模块逐步提升为真正的 agent

## 范围

### 做

- 只做 `PXR Activity Track`
- 只做小分子回归任务
- 只做本地运行
- 只做模板化 plan，不做开放式自动设计
- 只做少量基线模型
- 只做本地导出提交文件

### 不做

- 不做 `Structure Track`
- 不做自动提交 leaderboard
- 不做通用多任务 AutoML
- 不做复杂超参搜索
- 不做完整测试矩阵
- 不做多 agent 并行编排
- 不在 MVP 第一阶段就落地真实 7 agent 运行时

## 问题定义

### 任务类型

- `small_molecule_regression`

### 预测目标

- `pEC50`

### 主指标

- `RAE`

### 交付物

- `submission.csv`
- `final_report.json`
- 每个方案的结果文件

### 提交格式

最终提交文件必须包含：

- `SMILES`
- `Molecule Name`
- `pEC50`

## MVP 组件

### 1. Task Spec

把 challenge 问题写成结构化配置，而不是只用自然语言。

输出：

- `outputs/<run_id>/task_spec.json`

内容至少包括：

- challenge name
- track
- task type
- target column
- metric
- required submission columns
- data directory

### 2. Data Validator

负责检查数据是否齐全、列名是否符合预期、训练集和测试集是否能正常加载。

输出：

- `outputs/<run_id>/dataset_report.json`

### 3. Skill Registry + Fallback

先保留已有 skill registry 思路，但对这个 MVP 来说，重点是：

- 如果没有合适的 PXR small-molecule model skill
- 自动回退到通用 chemistry baseline

输出：

- `outputs/<run_id>/retrieval_result.json`

### 4. Template Planner

不让 planner 自由发挥，直接固定 3 个模板：

1. Morgan fingerprint + Ridge
2. Morgan fingerprint + LightGBM/XGBoost
3. RDKit descriptors + RandomForest/CatBoost

输出：

- `outputs/<run_id>/design_plans.json`

### 5. Experiment Runner

每个方案必须能实际运行并产出结构化结果。

输出目录建议：

- `outputs/<run_id>/plans/<plan_id>/`

每个 plan 至少产出：

- `train.py`
- `config.json`
- `metrics.json`
- `oof_predictions.csv`
- `test_predictions.csv`

### 6. Selector

比较所有方案，选出最佳方案。

输出：

- `outputs/<run_id>/leaderboard.json`
- `outputs/<run_id>/best_plan.json`

### 7. Tuner

对最佳 baseline 方案做最小调参，并重新训练与评估。

MVP 中只做小范围 tuning：

- Ridge 的 `alpha`
- RandomForest 的 `n_estimators` 和 `max_depth`
- 如果使用 GBDT，则只做很小搜索空间

输出：

- `outputs/<run_id>/tuning_trials.json`
- `outputs/<run_id>/tuning_summary.json`
- tuned result files

### 8. Submission Exporter

基于最佳方案，把测试集预测写成 challenge 要求的提交格式。

输出：

- `outputs/<run_id>/submission.csv`
- `outputs/<run_id>/final_report.json`

### 9. Minimal Orchestrator

先不用复杂 manager loop，MVP 只要一个顺序执行器：

1. parse task
2. validate data
3. retrieve or fallback
4. build plans
5. run baseline plans
6. select current best
7. run tuning on the best plan
8. select best overall result
9. export submission

## 目标 Agent 架构

未来目标拆分为 7 个 agent：

1. `Manager`
2. `Setup`
3. `Retrieval`
4. `Designer`
5. `Coder`
6. `Tuner`
7. `Exporter/Aggregator`

当前 MVP 对应关系：

- `Manager` -> `manager.py`
- `Setup` -> `schemas.py + data_utils.py`
- `Retrieval` -> `retrieval.py`
- `Designer` -> `planner.py`
- `Coder` -> `runner.py` 中的训练执行部分
- `Tuner` -> 后续单独的调参逻辑或 `tuner.py`
- `Exporter/Aggregator` -> `selector.py + exporter.py`

因此，MVP 的具体步骤需要围绕“按 7 agent 职责拆模块，但先单主控执行”来写。

## 推荐目录

```text
bio-model-skills-creator/
  manager.py
  plan_mvp.md
  implementation_steps.md
  data/
    pxr_activity/
  outputs/
  registry/
  skills/
  src/
    schemas.py
    data_utils.py
    retrieval.py
    planner.py
    runner.py
    tuner.py
    selector.py
    exporter.py
```

## MVP Definition of Done

满足以下条件即可认为 MVP 完成：

1. 可以从命令行启动一次完整运行
2. 能生成 `task_spec.json`
3. 能验证数据并输出 `dataset_report.json`
4. 能生成固定的 3 个 plan
5. 至少有 2 个 plan 可以成功训练和预测
6. 能输出 plan 级别指标
7. 能对最佳 baseline 方案做一轮最小 tuning
8. 能自动选出最佳 overall plan
9. 能导出符合格式的 `submission.csv`
10. 全部产物落在同一个 `run_id` 目录下

## 实施优先级

### P0

- `schemas.py`
- `data_utils.py`
- `planner.py`
- `runner.py`

### P1

- `tuner.py`
- `selector.py`
- `exporter.py`
- `manager.py` 串联

### P2

- `retrieval.py` 接入 skill registry
- registry 缓存
- 更细的日志和错误处理
