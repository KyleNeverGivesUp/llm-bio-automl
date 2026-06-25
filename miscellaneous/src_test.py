from src.constants import RUN_ID
from src.task_spec_builder import build_pxr_activity_task_spec, write_task_spec
from src.planner import build_mvp_plans, write_plans

run_id = RUN_ID
task_spec = build_pxr_activity_task_spec("data/pxr_activity")

write_task_spec(task_spec, f"outputs/{run_id}/task_spec.json")

plans = build_mvp_plans(task_spec, run_id)
write_plans(plans, f"outputs/{run_id}/design_plans.json")
