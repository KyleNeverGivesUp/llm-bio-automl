"""Generate and run tuning trials for the best baseline plan."""

import json
import os
import re
from dataclasses import asdict
from pathlib import Path

from anthropic import Anthropic

from src.agent.LLM_base import LLMJsonAgent
from src.agent.agent_context import AgentResult, RunContext
from src.data_utils import load_activity_test, load_activity_train
from src.schemas import PlanSpec
from src.tuner import (
    build_tuning_trials,
    find_base_plan,
    load_best_plan,
    load_design_plans,
    run_tuning_trials,
    write_tuning_summary,
    write_tuning_trials,
)


class TunerAgent(LLMJsonAgent):
    name = "tuner"

    def _best_known_baseline_plan(self) -> PlanSpec:
        return PlanSpec(
            plan_id="skill_1_chemberta-77m-mtr_ridge",
            name="DeepChem/ChemBERTa-77M-MTR Embedding + Ridge",
            feature_type="skill_embedding",
            model_type="ridge",
            params={
                "alpha": 1.0,
                "pooling": "cls",
                "max_length": 256,
                "batch_size": 16,
            },
            skill_ref="DeepChem/ChemBERTa-77M-MTR",
            skill_path="skills/models/chemistry/DeepChem--ChemBERTa-77M-MTR/SKILL.md",
            notes=(
                "Fixed tuning base. Strongest validated baseline around "
                "MAE 0.5537 / RAE 0.6085 / R2 0.5489."
            ),
        )

    def _anthropic_client(self) -> tuple[Anthropic, str]:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

        client_kwargs = {"api_key": api_key}
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
        if base_url:
            client_kwargs["base_url"] = base_url

        model = os.environ.get("ANTHROPIC_MODEL", "").strip()
        if not model:
            raise RuntimeError("ANTHROPIC_MODEL is not configured. Set it to your Sonnet 4.6 model id.")
        return Anthropic(**client_kwargs), model

    def _anthropic_response_text(self, response) -> str:
        parts: list[str] = []
        for block in response.content:
            text = getattr(block, "text", "")
            if text:
                parts.append(text)
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError("Anthropic returned empty text content.")
        return text

    def _parse_json_response(self, raw_text: str) -> dict:
        text = raw_text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced:
            text = fenced.group(1)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("No JSON object found in Anthropic response.")
        return json.loads(text[start : end + 1])

    def _call_anthropic_json_logged(
        self,
        context: RunContext,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        log_path = context.run_dir / "llm_logs" / f"{call_name}.json"
        client, model_name = self._anthropic_client()
        request_payload = {
            "provider": "anthropic",
            "model": model_name,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }
        log_payload = {
            "status": "started",
            "call_name": call_name,
            "request": request_payload,
            "response_json": None,
            "raw_text_content": None,
            "parsed_json": None,
            "error": None,
        }

        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=self.max_tokens,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = self._anthropic_response_text(response)
            parsed = self._parse_json_response(raw_text)

            log_payload["status"] = "ok"
            log_payload["response_json"] = response.model_dump() if hasattr(response, "model_dump") else None
            log_payload["raw_text_content"] = raw_text
            log_payload["parsed_json"] = parsed
            self._write_llm_log(log_path, log_payload)
            return parsed
        except Exception as exc:
            log_payload["status"] = "error"
            log_payload["error"] = str(exc)
            self._write_llm_log(log_path, log_payload)
            raise

    def _validate_trials(self, payload: dict, base_plan: PlanSpec) -> list[PlanSpec]:
        trials: list[PlanSpec] = []
        for idx, trial in enumerate(payload.get("trials", []), start=1):
            trial.setdefault("plan_id", f"{base_plan.plan_id}_draft_{idx}")
            trial.setdefault("name", f"{base_plan.name} Draft Tune {idx}")
            trial.setdefault("feature_type", base_plan.feature_type)
            trial.setdefault("model_type", base_plan.model_type)
            trial.setdefault("params", dict(base_plan.params))
            feature_type = trial.get("feature_type")
            if feature_type in {"skill_embedding", "skill_embedding_plus_morgan"}:
                trial["skill_ref"] = base_plan.skill_ref
                trial["skill_path"] = base_plan.skill_path
            else:
                trial["skill_ref"] = trial.get("skill_ref")
                trial["skill_path"] = trial.get("skill_path")
            plan = PlanSpec(**trial)
            if not self._is_allowed_trial(plan):
                raise ValueError(
                    f"Unsupported trial proposed by LLM: feature_type={plan.feature_type}, model_type={plan.model_type}"
                )
            trials.append(plan)
        return trials

    def _load_optional_json(self, path: Path) -> dict | list | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_fixed_baseline_metrics(self, run_dir: Path, base_plan: PlanSpec) -> dict:
        metrics_path = run_dir / "plans" / base_plan.plan_id / "metrics.json"
        if metrics_path.exists():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            "plan_id": base_plan.plan_id,
            "plan_name": base_plan.name,
            "feature_type": base_plan.feature_type,
            "model_type": base_plan.model_type,
            "skill_ref": base_plan.skill_ref,
        }

    def _load_prior_round_summaries(self, run_dir: Path, before_round: int) -> list[dict]:
        summaries: list[dict] = []
        for round_idx in range(1, before_round):
            path = run_dir / f"tuning_summary_round_{round_idx}.json"
            if path.exists():
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
        return summaries

    def _load_prior_round_reports(self, run_dir: Path, before_round: int) -> list[dict]:
        reports: list[dict] = []
        for round_idx in range(1, before_round):
            path = run_dir / f"tuner_report_round_{round_idx}.json"
            if path.exists():
                reports.append(json.loads(path.read_text(encoding="utf-8")))
        return reports

    def _recent_optimization_diagnostics(self, prior_round_summaries: list[dict]) -> str:
        if not prior_round_summaries:
            return "- No prior tuning rounds yet."

        recent = prior_round_summaries[-8:]
        lines: list[str] = []
        family_keys: list[str] = []
        raes: list[float] = []
        ridge_alphas: list[float] = []

        for idx, summary in enumerate(recent, start=max(1, len(prior_round_summaries) - len(recent) + 1)):
            best_tuned = summary.get("best_tuned") or {}
            feature_type = best_tuned.get("feature_type", "unknown")
            model_type = best_tuned.get("model_type", "unknown")
            family_key = f"{feature_type}::{model_type}"
            family_keys.append(family_key)

            trial_params = best_tuned.get("trial_params") or {}
            rae = best_tuned.get("rae")
            if isinstance(rae, (int, float)):
                raes.append(float(rae))

            if family_key == "skill_embedding::ridge":
                alpha = trial_params.get("alpha")
                if isinstance(alpha, (int, float)):
                    ridge_alphas.append(float(alpha))

            lines.append(
                f"- Round {idx}: family={family_key}, "
                f"rae={rae}, params={json.dumps(trial_params, ensure_ascii=False, sort_keys=True)}"
            )

        if len(raes) >= 3:
            recent_improvement = max(raes[:-1]) - min(raes)
            lines.append(
                f"- Diagnostic: recent best RAE improvement across the last {len(raes)} rounds is "
                f"{recent_improvement:.6f}."
            )
            if recent_improvement < 0.0015:
                lines.append(
                    "- Diagnostic: search appears close to saturated; avoid another tiny local tweak in the same "
                    "parameter direction unless you can justify a concrete hypothesis."
                )

        if len(family_keys) >= 4 and len(set(family_keys[-4:])) == 1:
            lines.append(
                f"- Diagnostic: the last 4 rounds all stayed in the same family ({family_keys[-1]}). "
                "If gains are small, switch strategy inside the allowed search space."
            )

        if len(ridge_alphas) >= 4 and max(ridge_alphas[-4:]) - min(ridge_alphas[-4:]) <= 0.1:
            lines.append(
                "- Diagnostic: recent ridge rounds have only nudged alpha in a very narrow band. "
                "Do not keep inching alpha by tiny steps if the metric barely moves."
            )

        return "\n".join(lines)

    def _allowed_param_block(self) -> str:
        lines = [
            '- Propose exactly 1 trial.',
            '- Only two model families are allowed for search:',
            '  1. feature_type="skill_embedding", model_type="ridge" using DeepChem/ChemBERTa-77M-MTR.',
            '  2. feature_type="skill_embedding_plus_morgan", model_type="xgboost" using DeepChem/ChemBERTa-77M-MTR + Morgan.',
            '- Do not propose elastic_net, random_forest, plain morgan_fingerprint, or any other backbone.',
            '- For the ridge family, allowed params are only:',
            '  - pooling: "cls" only',
            '  - max_length: 64, 96, 128',
            '  - batch_size: 16 or 32',
            '  - alpha: choose around the known strong region 0.8 to 1.0',
            '- For the fusion + xgboost family, allowed params are only:',
            '  - pooling: "cls" only',
            '  - max_length: 64, 96, 128',
            '  - batch_size: 16 or 32',
            '  - radius: 2 or 3',
            '  - n_bits: 1024, 2048, or 4096',
            '  - n_estimators: 200, 300, or 500',
            '  - max_depth: 4, 6, or 8',
            '  - learning_rate: 0.03, 0.05, or 0.08',
            '  - subsample: 0.8, 0.9, or 1.0',
            '  - colsample_bytree: 0.8, 0.9, or 1.0',
            '  - reg_alpha: 0.0, 0.1, or 0.3',
            '  - reg_lambda: 0.5, 1.0, or 2.0',
            '  - random_state: 42',
            '  - n_jobs: 1 or -1',
            '- Do not emit unsupported params.',
        ]
        return "\n".join(lines)

    def _is_allowed_trial(self, plan: PlanSpec) -> bool:
        return (
            (plan.feature_type == "skill_embedding" and plan.model_type == "ridge")
            or (plan.feature_type == "skill_embedding_plus_morgan" and plan.model_type == "xgboost")
        )

    def _namespace_trials(self, trials: list[PlanSpec], base_plan: PlanSpec, iteration: int) -> list[PlanSpec]:
        namespaced: list[PlanSpec] = []
        for idx, trial in enumerate(trials, start=1):
            trial_id_seed = trial.plan_id or f"trial_{idx}"
            trial.plan_id = f"{trial_id_seed}_r{iteration}"
            if not trial.name:
                trial.name = f"{base_plan.name} Round {iteration} Tune {idx}"
            else:
                trial.name = f"{trial.name} [Round {iteration}]"
            namespaced.append(trial)
        return namespaced

    def run(self, context: RunContext, iteration: int = 1, total_iterations: int = 5) -> AgentResult:
        design_plans = load_design_plans(context.run_dir / "design_plans.json")
        fixed_baseline = self._best_known_baseline_plan()
        try:
            base_plan = next(plan for plan in design_plans if plan.plan_id == fixed_baseline.plan_id)
        except StopIteration:
            base_plan = fixed_baseline

        current_best = load_best_plan(context.run_dir / "best_plan.json")
        llm_log_path = context.run_dir / "llm_logs" / f"tuner_round_{iteration}.json"
        leaderboard = self._load_optional_json(context.run_dir / "leaderboard.json")
        prior_round_summaries = self._load_prior_round_summaries(context.run_dir, before_round=iteration)
        prior_round_reports = self._load_prior_round_reports(context.run_dir, before_round=iteration)
        fixed_baseline_metrics = self._load_fixed_baseline_metrics(context.run_dir, base_plan)
        recent_diagnostics = self._recent_optimization_diagnostics(prior_round_summaries)

        system_prompt = (
            "You are Claude acting as a parameter optimization specialist for a chemistry "
            "AutoML system. The tuning base is fixed and must not change. "
            "After each round, use all previous results and process notes to propose the single next best trial. "
            "Prioritize optimization around the strongest-performing model families and parameter regions already observed. "
            "Detect when a local search pattern has become low-yield and deliberately change strategy instead of "
            "continuing meaningless micro-adjustments. "
            "Return JSON only."
        )
        user_prompt = f"""
Iteration:
- current_round = {iteration}
- total_rounds = {total_iterations}

Fixed tuning base plan that must remain unchanged:
{json.dumps(asdict(base_plan), indent=2, ensure_ascii=False)}

Current best plan across all executed runs:
{json.dumps(current_best, indent=2)}

Current leaderboard:
{json.dumps(leaderboard, indent=2, ensure_ascii=False)}

Prior tuning summaries:
{json.dumps(prior_round_summaries, indent=2, ensure_ascii=False)}

Prior tuner process reports:
{json.dumps(prior_round_reports, indent=2, ensure_ascii=False)}

Recent optimization diagnostics derived from prior rounds:
{recent_diagnostics}

Optimization instructions:
{self._allowed_param_block()}
- Use the current best metrics, all previous trial outcomes, and prior process notes to propose the single next trial.
- This is sequential optimization. Do not repeat previous parameter settings unless you have a strong reason.
- The 0.55 baseline is the anchor and reference point, but you may choose the next model family and all parameters yourself.
- For the proposed trial, you choose the model, feature type, and all parameter values yourself.
- If using skill embeddings, consider backbone choice, pooling, max_length, and batch_size jointly with model params.
- Prefer exploiting the best-performing model families already visible in the leaderboard before exploring weaker families.
- Give more search budget to the hybrid family: skill_embedding_plus_morgan + xgboost.
- Use the ridge family mainly as a strong control/baseline or for narrow local refinement around its known good region.
- Prefer local refinement around the current top results: nearby hyperparameters, nearby feature settings, and nearby model families.
- Only switch away from the currently strongest family if you can justify a concrete recovery hypothesis from the prior results.
- Treat the top-ranked models in the leaderboard as the primary search region.
- If recent rounds show minimal gain from repeatedly nudging one parameter, stop that pattern and try a different
  hypothesis within the allowed search space.
- Do not behave like a blind grid search. When a mode looks saturated, change family or change a more meaningful
  feature block such as max_length, radius, n_bits, depth, learning_rate, or sample/column subsampling.
- Avoid proposing another tiny ridge alpha tweak when the last several ridge-alpha tweaks have produced negligible gain.
- Prefer larger, hypothesis-driven moves over tiny parameter drifts once improvement falls below a meaningful threshold.

Return JSON exactly in this shape:
{{
  "summary": "<short summary>",
  "trials": [
    {{
      "plan_id": "<trial id>",
      "name": "<trial name>",
      "feature_type": "morgan_fingerprint or skill_embedding or skill_embedding_plus_morgan",
      "model_type": "ridge or elastic_net or random_forest or xgboost",
      "params": {{...}},
      "skill_ref": "<skill ref or null>",
      "skill_path": "<skill path or null>",
      "notes": "<short note>"
    }}
  ]
}}
        """

        used_fallback = False
        try:
            payload = self._call_anthropic_json_logged(context, f"tuner_round_{iteration}", system_prompt, user_prompt)
            trials = self._validate_trials(payload, base_plan)
            if len(trials) != 1:
                raise ValueError(f"Expected exactly 1 tuning trial, got {len(trials)}.")
            trials = self._namespace_trials(trials, base_plan, iteration)
        except Exception as exc:
            used_fallback = True
            payload = {
                "summary": "Fallback to deterministic tuning grid.",
                "fallback_error": str(exc),
            }
            fallback_trials = build_tuning_trials(base_plan)
            selected_fallback = fallback_trials[(iteration - 1) % len(fallback_trials)]
            trials = self._namespace_trials([selected_fallback], base_plan, iteration)

        round_trials_path = context.run_dir / f"tuning_trials_round_{iteration}.json"
        write_tuning_trials(trials, round_trials_path)
        write_tuning_trials(trials, context.run_dir / "tuning_trials.json")

        train_df = load_activity_train(context.data_dir)
        test_df = load_activity_test(context.data_dir)
        tuned_results = run_tuning_trials(trials, train_df, test_df, context.run_dir)

        round_summary_path = context.run_dir / f"tuning_summary_round_{iteration}.json"
        write_tuning_summary(
            base_plan=base_plan,
            best_baseline=fixed_baseline_metrics,
            tuned_results=tuned_results,
            output_path=round_summary_path,
            primary_metric=context.primary_metric,
        )
        write_tuning_summary(
            base_plan=base_plan,
            best_baseline=fixed_baseline_metrics,
            tuned_results=tuned_results,
            output_path=context.run_dir / "tuning_summary.json",
            primary_metric=context.primary_metric,
        )

        report_path = context.run_dir / f"tuner_report_round_{iteration}.json"
        report_path.write_text(
            json.dumps(
                {
                    **payload,
                    "iteration": iteration,
                    "total_iterations": total_iterations,
                    "used_fallback": used_fallback,
                    "llm_log_path": str(llm_log_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (context.run_dir / "tuner_report.json").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

        return AgentResult(
            agent_name=self.name,
            status="done",
            summary=f"Completed LLM-guided tuning trials for round {iteration}/{total_iterations}.",
            outputs={
                "iteration": iteration,
                "tuning_trials_path": str(round_trials_path),
                "latest_tuning_trials_path": str(context.run_dir / "tuning_trials.json"),
                "tuning_summary_path": str(round_summary_path),
                "latest_tuning_summary_path": str(context.run_dir / "tuning_summary.json"),
                "tuner_report_path": str(report_path),
                "used_fallback": used_fallback,
                "llm_log_path": str(llm_log_path),
            },
        )
