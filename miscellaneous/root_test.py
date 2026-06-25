import sys
import json
import shutil
import subprocess


def hf_cli_command() -> list[str]:
    hf = shutil.which("hf")
    if hf:
        return [hf]

    legacy = shutil.which("huggingface-cli")
    if legacy:
        return [legacy]

    raise RuntimeError("未找到 Hugging Face CLI，请先安装 `huggingface_hub`。")


def run_hf_json(args: list[str]) -> dict | list | None:
    command = hf_cli_command() + args + ["--format", "json"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "hf CLI 调用失败")
    stdout = result.stdout.strip()
    if not stdout:
        return None
    return json.loads(stdout)


def get_hf_model_info(model_id: str) -> dict:
    model = run_hf_json([
        "models",
        "info",
        model_id,
        "--expand",
        "downloads,likes,trendingScore,spaces,evalResults,cardData,tags,pipeline_tag",
    ]) or {}

    collections = run_hf_json([
        "collections",
        "list",
        "--item",
        f"models/{model_id}",
        "--sort",
        "upvotes",
        "--limit",
        "5",
    ]) or []

    if isinstance(collections, list):
        model["collectionSlugs"] = [
            item.get("slug", "")
            for item in collections
            if isinstance(item, dict) and item.get("slug")
        ]
    else:
        model["collectionSlugs"] = []

    return model


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "facebook/esm2_t6_8M_UR50D"

    try:
        model = get_hf_model_info(model_id)
    except RuntimeError as exc:
        print(str(exc))
        return

    downloads = model.get("downloads")
    likes = model.get("likes")
    trending_score = model.get("trendingScore")
    spaces = model.get("spaces") or []
    eval_results = model.get("evalResults") or []
    card_data = model.get("cardData") or {}
    collection_slugs = model.get("collectionSlugs") or []

    print(f"模型: {model_id}")
    print(f"下载量: {downloads}")
    print(f"点赞数: {likes}")
    print(f"Trending Score: {trending_score}")
    print(f"Spaces 数量: {len(spaces)}")
    print(f"Eval Results 数量: {len(eval_results)}")
    print(f"Collection 收录数: {len(collection_slugs)}")
    print(f"Collections: {collection_slugs}")
    print(f"Card Data 字段: {sorted(card_data.keys()) if isinstance(card_data, dict) else []}")


if __name__ == "__main__":
    main()
