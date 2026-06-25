import json
from pathlib import Path


def load_metrics_file(metrics_path: str) -> dict:
    return json.loads(Path(metrics_path).read_text(encoding="utf-8"))


def load_all_metrics(run_dir: str) -> list[dict]:
    run_path = Path(run_dir)
    metrics_files = sorted(run_path.glob("plans/*/metrics.json"))

    metrics_list = []
    for metrics_file in metrics_files:
        metrics = load_metrics_file(str(metrics_file))
        metrics["metrics_path"] = str(metrics_file)
        metrics_list.append(metrics)

    return metrics_list


def sort_metrics(metrics_list: list[dict], primary_metric: str) -> list[dict]:
    if not metrics_list:
        return []

    if primary_metric == "RAE":
        sort_key = "rae"
        reverse = False
    elif primary_metric in {"mae"}:
        sort_key = primary_metric
        reverse = False
    elif primary_metric in {"r2"}:
        sort_key = primary_metric
        reverse = True
    else:
        sort_key = "mae"
        reverse = False

    valid_metrics = [m for m in metrics_list if sort_key in m]
    return sorted(valid_metrics, key=lambda x: x[sort_key], reverse=reverse)


def select_best_plan(metrics_list: list[dict], primary_metric: str) -> dict:
    ranked = sort_metrics(metrics_list, primary_metric)
    if not ranked:
        raise ValueError("No valid metrics found to select best plan.")

    return ranked[0]


def write_leaderboard(metrics_list: list[dict], output_path: str, primary_metric: str) -> None:
    ranked = sort_metrics(metrics_list, primary_metric)

    leaderboard = {
        "primary_metric": primary_metric,
        "ranking_metric_used": "rae" if primary_metric == "RAE" else primary_metric,
        "n_runs": len(ranked),
        "results": ranked,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(leaderboard, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_best_plan(best_plan: dict, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(best_plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
