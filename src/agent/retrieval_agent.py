"""Choose the retrieval strategy for the current task."""

import json
import re
from pathlib import Path


from src.agent.LLM_base import LLMJsonAgent
from src.agent.agent_context import AgentResult, RunContext
from src.schemas import TaskInference


class RetrievalAgent(LLMJsonAgent):
    name = "retrieval"

    def _infer_task_heuristic(self, task_spec: dict) -> TaskInference:
        title = str(task_spec.get("task_title") or "")
        description = str(task_spec.get("task_description") or "")
        target_column = str(task_spec.get("target_column") or "")
        primary_metric = str(task_spec.get("primary_metric") or "")
        submission_columns = " ".join(
            str(column) for column in (task_spec.get("submission_columns") or [])
        )
        text = " ".join(
            [title, description, target_column, primary_metric, submission_columns]
        ).lower()

        if "smiles" in text or "molecule" in text or "pec50" in text:
            return TaskInference(
                task_domain="chemistry",
                task_modality="smiles",
                task_type="regression",
                reason=(
                    "Heuristic fallback inferred a chemistry regression task because the "
                    "task references SMILES or molecules and predicts a continuous activity value."
                ),
            )

        if "protein" in text and ("sequence" in text or "structure" in text):
            return TaskInference(
                task_domain="protein",
                task_modality="protein_sequence" if "sequence" in text else "protein_structure",
                task_type="regression" if "regression" in text else "classification",
                reason="Heuristic fallback inferred a protein task from the task description.",
            )

        if "dna" in text or "rna" in text or "genomic" in text:
            return TaskInference(
                task_domain="genomics",
                task_modality="genomics_sequence",
                task_type="regression" if "regression" in text else "classification",
                reason="Heuristic fallback inferred a genomics task from the task description.",
            )

        return TaskInference(
            task_domain="chemistry",
            task_modality="smiles",
            task_type="regression",
            reason=(
                "Heuristic fallback defaulted to chemistry regression because the project "
                "currently targets small-molecule property prediction."
            ),
        )

    def _extract_description(self, skill_text: str) -> str:
        front_matter_match = re.search(
            r"^---\s*\n(.*?)\n---\s*\n",
            skill_text,
            flags=re.DOTALL,
        )
        if front_matter_match:
            front_matter = front_matter_match.group(1)
            desc_match = re.search(r'^description:\s*(.+)$', front_matter, flags=re.MULTILINE)
            if desc_match:
                return desc_match.group(1).strip().strip('"').strip("'")

        for line in skill_text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped[:300]

        return ""

    def _load_skill_catalog(self) -> list[dict]:
        manifest_path = Path("skills/models/manifest.json")
        if not manifest_path.exists():
            return []

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        catalog: list[dict] = []

        for item in manifest.get("skills", []):
            skill_path = Path(item["path"])
            description = ""
            if skill_path.exists():
                description = self._extract_description(skill_path.read_text(encoding="utf-8"))

            catalog.append(
                {
                    "ref": item["ref"],
                    "domain": item["domain"],
                    "path": str(skill_path),
                    "description": description,
                }
            )
        
        return catalog

    def _infer_task(self, context: RunContext, task_spec: dict, available_domains: list[str]) -> TaskInference:
        system_prompt = (
    "You are an expert in biology, drug discovery, and machine learning, "
    "helping build an AutoML system. "
    "Infer the task domain, task modality, and task type from the task title "
    "and task description. "
    "Define task_domain by the primary modeling object and model family that "
    "should be used for prediction, not merely by the biological target "
    "mentioned in the task. "
    "Use 'chemistry' when the main predictive object is a compound, ligand, "
    "small molecule, SMILES string, molecular graph, or molecular descriptor. "
    "Use 'protein' only when the main predictive object is a protein sequence, "
    "protein structure, or protein-specific representation. "
    "Use 'genomics' only when the main predictive object is DNA, RNA, or other "
    "genomic sequence representations. "
    "Return JSON only."
)


        user_prompt = f"""
Task spec:
{json.dumps(task_spec, indent=2, ensure_ascii=False)}

Available local model-skill domains:
{json.dumps(available_domains, indent=2, ensure_ascii=False)}

Classification guidance:
- If the task predicts properties, activities, affinities, or other outcomes for compounds, ligands, or SMILES-based molecules, set task_domain = "chemistry".
- If the task predicts properties or functions from protein sequences or protein structures, set task_domain = "protein".
- If the task predicts properties or functions from DNA or RNA sequences, set task_domain = "genomics".
- Do not assign task_domain based only on the biological target. For example, predicting a compound's activity against a protein receptor is still a chemistry task if the predictive input is a small molecule.
- task_modality should describe the primary modeling input, such as small_molecule, smiles, molecular_graph, protein_sequence, protein_structure, genomics_sequence, or biomedical_text.
- task_type should describe the machine learning objective, such as regression, classification, ranking, sequence_labeling, or structure_prediction.

Respond with JSON exactly in this shape:
{{
  "task_domain": "<domain>",
  "task_modality": "<modality>",
  "task_type": "<task type>",
  "reason": "<short explanation>"
}}
"""
        
        payload = self.call_json_logged(context, "retrieval_infer", system_prompt, user_prompt)
        return TaskInference(**payload)
    
    def _select_skills(
        self,
        context: RunContext,
        task_spec: dict,
        task_inference: TaskInference,
        candidate_skills: list[dict],    
    ) -> dict:
        if task_inference.task_domain == "chemistry":
            hardcoded_refs = [
                "DeepChem/ChemBERTa-77M-MTR",
                "DeepChem/ChemBERTa-100M-MLM",
                "DeepChem/ChemBERTa-10M-MTR",
            ]
            selected_skills = [
                {
                    "ref": item["ref"],
                    "domain": item["domain"],
                    "path": item["path"],
                    "reason": (
                        "Fixed chemistry retrieval candidate for controlled downstream testing."
                    ),
                }
                for item in candidate_skills
                if item["ref"] in hardcoded_refs
            ]

            if len(selected_skills) != len(hardcoded_refs):
                missing = sorted(set(hardcoded_refs) - {item["ref"] for item in selected_skills})
                raise ValueError(f"Missing hardcoded chemistry skills: {missing}")

            return {
                "mode": "retrieved",
                "selected_strategy": "model_skills",
                "selected_skills": selected_skills,
                "reason": (
                    "Using a fixed set of chemistry model skills to isolate the impact of "
                    "retrieved upstream model candidates on downstream pipeline performance."
                ),
            }

        system_prompt = (
            "You are a RetrievalAgent for a biological ML AutoML system."
            "Given a task spec, inferred task attributes, and local model-skill candidates, "
            "select the 3 most relevant model skills. Return JSON only."
        )

        user_prompt = f"""
Task spec:
{json.dumps(task_inference.__dict__, indent=2, ensure_ascii=False)}

Candidate local model skills:
{json.dumps(candidate_skills, indent=2, ensure_ascii=False)}

Select exactly 3 skills if possible, otherwise select as many relevant skills as available.

Respond with JSON exactly in this shape:
{{
  "mode": "retrieved",
  "selected_strategy": "model_skills",
  "selected_skills": [
    {{
      "ref": "<skill ref>",
      "domain": "<domain>",
      "path": "<path>",
      "reason": "<short reason>"
    }}
  ],
  "reason": "<overall reason>"
}}
"""
        
        return self.call_json_logged(context, "retrieval_select", system_prompt, user_prompt)
        

    def _fallback_decision(
        self,
        skill_catalog: list[dict],
        task_inference: dict | None,
        reason: str,
        fallback_error: str | None = None,
    ) -> dict:
        return {
            "mode": "fallback",
            "selected_strategy": "chemistry_baselines",
            "task_inference": task_inference,
            "selected_skills": [],
            "n_total_skills": len(skill_catalog),
            "available_domains": sorted({item["domain"] for item in skill_catalog}),
            "reason": reason,
            "fallback_error": fallback_error,
        }

    def run(self, context: RunContext) -> AgentResult:
        task_spec = json.loads((context.run_dir / "task_spec.json").read_text(encoding="utf-8"))
        llm_log_path = context.run_dir / "llm_logs" / "retrieval_select.json"

        skill_catalog = self._load_skill_catalog()
        available_domains = sorted({item["domain"] for item in skill_catalog})

        if not skill_catalog:
            decision = self._fallback_decision(
                skill_catalog=skill_catalog,
                task_inference=None,
                reason="No local model skills manifest found. Falling back to chemistry baselines.",
            )
            used_fallback = True
        else:
            try:
                task_inference = self._infer_task(context, task_spec, available_domains)
                print(task_inference)
            except Exception as exc:
                task_inference = self._infer_task_heuristic(task_spec)
                matched_skills = [
                    item for item in skill_catalog if item["domain"] == task_inference.task_domain
                ]

                if not matched_skills:
                    decision = self._fallback_decision(
                        skill_catalog=skill_catalog,
                        task_inference=task_inference.__dict__,
                        reason=(
                            "Task inference LLM call failed, and heuristic task inference matched "
                            "no local model skills. Falling back to chemistry baselines."
                        ),
                        fallback_error=str(exc),
                    )
                    used_fallback = True
                else:
                    try:
                        decision = self._select_skills(
                            context=context,
                            task_spec=task_spec,
                            task_inference=task_inference,
                            candidate_skills=matched_skills,
                        )

                        validated_skills = []
                        for item in matched_skills:
                            if item["ref"] in {selected["ref"] for selected in decision.get("selected_skills", [])}:
                                selected_reason = next(
                                    (
                                        selected.get("reason", "")
                                        for selected in decision.get("selected_skills", [])
                                        if selected.get("ref") == item["ref"]
                                    ),
                                    "",
                                )
                                validated_skills.append(
                                    {
                                        "ref": item["ref"],
                                        "domain": item["domain"],
                                        "path": item["path"],
                                        "description": item["description"],
                                        "reason": selected_reason,
                                    }
                                )

                        if not validated_skills:
                            raise ValueError("Heuristic task inference returned no valid selected skills.")

                        decision["task_inference"] = task_inference.__dict__
                        decision["selected_skills"] = validated_skills
                        decision["n_total_skills"] = len(skill_catalog)
                        decision["available_domains"] = available_domains
                        decision["reason"] = (
                            f"{decision.get('reason', '')} Heuristic task inference was used after "
                            f"the LLM call failed: {exc}"
                        ).strip()
                        used_fallback = False
                    except Exception as select_exc:
                        decision = self._fallback_decision(
                            skill_catalog=skill_catalog,
                            task_inference=task_inference.__dict__,
                            reason=(
                                "Task inference LLM call failed, and heuristic skill retrieval "
                                "also failed. Falling back to chemistry baselines."
                            ),
                            fallback_error=f"infer_error={exc}; select_error={select_exc}",
                        )
                        used_fallback = True
            else:
                matched_skills = [
                    item for item in skill_catalog if item["domain"] == task_inference.task_domain
                ]

                if not matched_skills:
                    decision = self._fallback_decision(
                        skill_catalog=skill_catalog,
                        task_inference=task_inference.__dict__,
                        reason=(
                            "No local model skills matched the inferred task domain. "
                            "Falling back to chemistry baselines."
                        ),
                    )
                    used_fallback = True
                else:
                    try:
                        decision = self._select_skills(
                            context=context,
                            task_spec=task_spec,
                            task_inference=task_inference,
                            candidate_skills=matched_skills,
                        )

                        selected_refs = {
                            item["ref"] for item in decision.get("selected_skills", [])
                        }

                        validated_skills = []
                        for item in matched_skills:
                            if item["ref"] in selected_refs:
                                selected_reason = next(
                                    (
                                        selected.get("reason", "")
                                        for selected in decision.get("selected_skills", [])
                                        if selected.get("ref") == item["ref"]
                                    ),
                                    "",
                                )
                                validated_skills.append(
                                    {
                                        "ref": item["ref"],
                                        "domain": item["domain"],
                                        "path": item["path"],
                                        "description": item["description"],
                                        "reason": selected_reason,
                                    }
                                )

                        if not validated_skills:
                            raise ValueError("LLM returned no valid selected skills.")

                        decision["task_inference"] = task_inference.__dict__
                        decision["selected_skills"] = validated_skills
                        decision["n_total_skills"] = len(skill_catalog)
                        decision["available_domains"] = available_domains
                        used_fallback = False

                    except Exception as exc:
                        decision = self._fallback_decision(
                            skill_catalog=skill_catalog,
                            task_inference=task_inference.__dict__,
                            reason="Skill-based retrieval failed. Falling back to chemistry baselines.",
                            fallback_error=str(exc),
                        )
                        used_fallback = True

        output_path = context.run_dir / "retrieval_result.json"
        output_path.write_text(
            json.dumps(decision, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return AgentResult(
            agent_name=self.name,
            status="done",
            summary="Produced retrieval decision.",
            outputs={
                "retrieval_result_path": str(output_path),
                "decision": decision,
                "used_fallback": used_fallback,
                "llm_log_path": str(llm_log_path),
            },
        )            
