# AIBuildAI — reference architecture (our north star for the agent loop)

> **What this is.** AIBuildAI (Zhang, Qin, Cao, Zhang, Xie — 2026; arXiv 2604.14455) is the AI agent that
> ranked **#1 on OpenAI MLE-Bench**. It is the reference design for our LLM-orchestrated pipeline. This file
> records its architecture **from the authoritative source** — the official `workflow.png` diagram and README
> in `/Users/kyle/Projects/AI-Build-AI/` — so we stop mis-citing it from memory.
>
> **Correction (2026-06-26):** earlier notes called AIBuildAI "a manager + 3 sub-agents (designer/coder/tuner)."
> The official diagram is clearer: it is **6 named agents** — Setup, Manager, Designer, Coder, Tuner, Aggregator —
> **all LLM-driven** (Claude Opus 4.6 underneath). The "3" was the paper abstract's emphasis on the core
> modeling loop; the full pipeline has 6.

---

## 1. Overall flow (from the official workflow.png)

```
Input Task
  │
  ├─►【Setup Agent】 env config & data loading
  │
  ├─►【Iterative Multi-agent Loop】
  │      ┌─────────────────────────────────────────────┐
  │      │  Manager Agent  (coordinate & judge trajectories)
  │      │   Manager picks ONE of:                       │
  │      │     • Call next agent                         │
  │      │     • Finish & save model                     │
  │      │     • Stop (if bad result)                    │
  │      │                                               │
  │      │   Parallel candidate runs:                    │
  │      │     D₁ → C₁ → T₁ → … ✓                         │
  │      │     D₂ → C₂ → T₂ → … ✗                         │
  │      └─────────────────────────────────────────────┘
  │           each chain = Designer → Coder → Tuner
  │
  ├─►【Aggregator Agent】 ensemble & select best submission
  │
  └─► Final Submission
```

The Manager's decision rules (top of the diagram) list the callable roles as:
**setup · designer · coder · tuner · reviser · judge · stop** — so `reviser` and `judge` are additional
roles the Manager can invoke beyond the six boxes drawn.

---

## 2. Each agent — role · input · output · **driver**

| Agent | Role | Input | Output | Driver |
|---|---|---|---|---|
| **Setup** | env config & data loading | competition info, env specs, instructions to prep data/conda/dirs | setup report (data/code paths, packages installed, status) | 🤖 LLM |
| **Manager** | coordinate, judge candidate chains, decide next | context, workflow state, decision rules for the next agent | **one action** (call agent X / finish & save / stop) + reason + instructions | 🤖 LLM |
| **Designer** | search / revise method plans | context, dataset info, metric, available packages | report JSON: candidate plans (architecture, data, loss, training config, validation strategy) | 🤖 LLM |
| **Coder** | **implement the designed pipeline** | context, a designer's plan, instructions to implement | report JSON: **writes `config.py` + `train.py`**, runs a sanity check, artifacts, submission, best checkpoint | 🤖 LLM |
| **Tuner** | monitor the long run & tune hyperparameters | context, the candidate model's options | report JSON: changes made, results, previous results for comparison, within-budget confirmation | 🤖 LLM |
| **Aggregator** | ensemble & select best | context, all candidate code/data paths | report JSON: ensemble + TTA + retrain on full data, standalone `submit.py` | 🤖 LLM |

**All six are LLM-driven** — the diagram shows a single "Claude Opus 4.6" banner under Designer/Coder/Tuner
(and the Manager), i.e. one LLM drives every agent. CLI knobs that shape the loop:
`--num-candidates` (parallel chains), `--max-agent-calls` (Manager call budget), `--run-budget-minutes` /
`--pipeline-budget-minutes` (time budget), `--model` (the LLM).

---

## 3. Where we stand vs AIBuildAI (the architecture-B blueprint)

Our project's original six skills (`setup/designer/coder/tuner/aggregator/models`) were modelled on
AIBuildAI's six agents — but most were never actually wired to an LLM. Per-node comparison:

| Node | AIBuildAI | **Ours (today)** | Gap to close |
|---|---|---|---|
| **Setup** | LLM (env + data prep) | deterministic code (`data_io`, `curation`) | low value to LLM-ify; keep deterministic |
| **Manager** | **LLM decision loop** + parallel candidate chains + ✓/✗ judging | **LLM decision loop** (`skill_manager`, arch B) — **but no parallel chains** | add parallel candidate trajectories |
| **Designer** | LLM (proposes N plans) | ✅ LLM (`menu_designer` + `finetune_designer`) | aligned |
| **Coder** | **LLM writes `config.py`+`train.py`** + sanity check | **template/executor** (`cv_runner`, `finetune_runner`) — picks from fixed templates, does **not** write code | ⭐ **biggest gap — free codegen (P3)** |
| **Tuner** | LLM (tunes + full train) | ✅ LLM (`menu_tuner`) | aligned |
| **Aggregator** | LLM (ensemble + TTA + refit) | deterministic ridge stack (`aggregator` + `analog_judge`) | could LLM-orchestrate ensemble choice; low value |
| (retrieval) | — (implicit in setup/coder) | live HF + frontier (`hf_retrieval`) — our addition | n/a |

### What this means for architecture B
- **Closest already:** Designer, Tuner, and the Manager decision loop (arch B) — LLM-driven, like AIBuildAI.
- **Two real gaps to be "fully AIBuildAI-style":**
  1. **Parallel candidate chains** (D→C→T run as several competing trajectories the Manager judges ✓/✗).
     Ours runs decisions sequentially with no competing chains.
  2. **Coder writes code** (free codegen + sanity check), not just instantiating fixed templates. This is
     the deepest gap (our boundary is *template-based codegen*: the LLM picks WHAT to fine-tune from a fixed
     `{chemeleon, unimol}` list; the training code is a verified template per backbone).
- **Deliberate divergences (not gaps):** we keep Setup / Aggregator / judge deterministic on purpose —
  reproducibility and leak-safety matter more than LLM-ifying them for a single competition, and the Set-1
  judge must stay a fixed, trustworthy reward signal.

---

## 4. Source

- Diagram + prose: `/Users/kyle/Projects/AI-Build-AI/assets/workflow.png` and `README.md`.
- Citation:
  ```bibtex
  @misc{zhang2026aibuildai,
    title={AIBuildAI: An AI agent that automatically builds AI models},
    author={Ruiyi Zhang and Peijia Qin and Qi Cao and Li Zhang and Pengtao Xie},
    year={2026}
  }
  ```
