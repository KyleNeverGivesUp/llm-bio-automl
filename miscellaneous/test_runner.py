import logging

import pandas as pd

from src.constants import RUN_ID
from src.planner import build_mvp_plans
from src.runner import run_plan
from src.task_spec_builder import build_pxr_activity_task_spec


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


run_id = RUN_ID
logger.info("Starting test_runner with run_id=%s", run_id)

task_spec = build_pxr_activity_task_spec("data/pxr_activity")
logger.info("Built task spec for challenge=%s track=%s", task_spec.challenge_name, task_spec.track)

train_df = pd.read_csv("data/pxr_activity/train.csv")
test_df = pd.read_csv("data/pxr_activity/test.csv")
logger.info("Loaded datasets: train_rows=%d test_rows=%d", len(train_df), len(test_df))

plans = build_mvp_plans(task_spec, run_id)
logger.info("Built %d plans", len(plans))

supported_plans = [
    plan
    for plan in plans
    if plan.feature_type == "morgan_fingerprint" and plan.model_type in {"ridge", "random_forest"}
]
logger.info("Selected %d supported plans for Version 1 runner", len(supported_plans))

for plan in supported_plans:
    output_dir = f"outputs/{run_id}/plans/{plan.plan_id}"
    logger.info(
        "Running plan_id=%s feature_type=%s model_type=%s output_dir=%s",
        plan.plan_id,
        plan.feature_type,
        plan.model_type,
        output_dir,
    )
    metrics = run_plan(plan, train_df, test_df, output_dir)
    logger.info(
        "Finished plan_id=%s mae=%.6f r2=%.6f",
        plan.plan_id,
        metrics["mae"],
        metrics["r2"],
    )

logger.info("test_runner completed successfully")
