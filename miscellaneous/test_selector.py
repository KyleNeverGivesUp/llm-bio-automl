from src.constants import RUN_ID
from src.selector import load_all_metrics, select_best_plan, sort_metrics, write_best_plan, write_leaderboard

run_id = RUN_ID
run_dir = f"outputs/{run_id}"

metrics_list = load_all_metrics(run_dir)
best_plan = select_best_plan(metrics_list, primary_metric="RAE")
ranked_metrics = sort_metrics(metrics_list, primary_metric="RAE")

write_leaderboard(metrics_list, f"{run_dir}/leaderboard.json", primary_metric="RAE")
write_best_plan(best_plan, f"{run_dir}/best_plan.json")

print(f"run_id: {run_id}")
print("best overall plan:")
print(f"  plan_id: {best_plan['plan_id']}")
print(f"  plan_name: {best_plan['plan_name']}")
print(f"  model_type: {best_plan['model_type']}")
print(f"  feature_type: {best_plan['feature_type']}")
print(f"  mae: {best_plan['mae']}")
print(f"  r2: {best_plan['r2']}")

print("\nranking summary:")
for idx, metrics in enumerate(ranked_metrics, start=1):
    print(
        f"  {idx}. {metrics['plan_id']} | "
        f"mae={metrics.get('mae')} | r2={metrics.get('r2')}"
    )
