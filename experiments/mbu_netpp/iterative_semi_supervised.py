from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.mbu_netpp.common import ensure_dir, load_yaml, save_json, seed_everything
from experiments.mbu_netpp.semi_utils import (
    append_pseudo_pool,
    build_iteration_training_root,
    build_unlabeled_items,
    clone_config,
    export_active_learning_selection,
    filter_items_by_stems,
    infer_unlabeled_items,
    load_dataset_manifest,
    load_pseudo_pool,
    load_query_pending,
    save_csv,
    save_query_pending,
    save_stem_list,
    select_active_learning_candidates,
    select_pseudo_candidates,
    summarize_iteration,
)
from experiments.mbu_netpp.train import maybe_prepare_data, train_one_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="少量标注 + 伪标签 + 主动学习 的迭代式半监督训练")
    parser.add_argument("--config", required=True, help="半监督 YAML 配置路径")
    parser.add_argument("--fold", type=int, default=None, help="仅运行指定 fold")
    parser.add_argument("--run-all-folds", action="store_true", help="对全部 fold 执行迭代流程")
    return parser.parse_args()


def load_state(state_path: str | Path) -> dict[str, Any]:
    path = Path(state_path)
    if not path.exists():
        return {
            "completed_iterations": 0,
            "current_checkpoint": "",
            "supervised_checkpoint": "",
            "iteration_summaries": [],
            "reviewed_source_stems": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(state_path: str | Path, state: dict[str, Any]) -> None:
    save_json(state_path, state)


def resolve_fold_indices(config: dict[str, Any], args: argparse.Namespace) -> list[int]:
    if args.run_all_folds:
        return list(range(int(config["data"].get("num_folds", 3))))
    if args.fold is not None:
        return [int(args.fold)]
    return [0]


def build_iteration_train_config(
    base_config: dict[str, Any],
    prepared_root: str,
    output_root: str,
    teacher_checkpoint: str,
    retrain_cfg: dict[str, Any],
    iteration_index: int,
    teacher_fold_index: int,
) -> dict[str, Any]:
    config = clone_config(base_config)
    config["experiment"]["name"] = f"{base_config['experiment']['name']}_semi_f{teacher_fold_index}_iter{iteration_index:02d}"
    config["experiment"]["output_root"] = output_root
    config["data"]["prepared_root"] = prepared_root
    config["data"]["fold_manifest_name"] = "folds_1_seed42.json"
    config["data"]["num_folds"] = 1
    config["data"]["auto_prepare"] = False
    config["data"]["force_prepare"] = False
    config["training"]["epochs"] = int(retrain_cfg.get("epochs", config["training"].get("epochs", 20)))
    config["training"]["learning_rate"] = float(
        retrain_cfg.get("learning_rate", config["training"].get("learning_rate", 1e-4))
    )
    config["training"]["init_checkpoint"] = teacher_checkpoint
    config["training"]["init_checkpoint_template"] = ""
    return config


def get_base_labeled_stems(prepared_root: str | Path) -> set[str]:
    dataset_manifest = load_dataset_manifest(prepared_root)
    return {str(item["stem"]) for item in dataset_manifest.get("items", [])}


def get_reviewed_labeled_stems(reviewed_roots: list[str]) -> set[str]:
    stems: set[str] = set()
    for root in reviewed_roots:
        dataset_manifest = load_dataset_manifest(root)
        stems.update({str(item["stem"]) for item in dataset_manifest.get("items", [])})
    return stems


def filter_query_pending_against_reviewed(
    pending_items: list[dict[str, Any]],
    reviewed_stems: set[str],
) -> list[dict[str, Any]]:
    return [
        item
        for item in pending_items
        if str(item.get("source_stem", item.get("stem"))) not in reviewed_stems
    ]


def resolve_initial_checkpoint(
    config: dict[str, Any],
    fold_index: int,
    fold_workspace: Path,
    state: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    if str(state.get("current_checkpoint", "")).strip():
        return str(state["current_checkpoint"]), None

    semi_cfg = config["semi_supervised"]
    teacher_template = str(semi_cfg.get("teacher_checkpoint_template", "")).strip()
    if teacher_template:
        checkpoint_path = teacher_template.format(fold_index=fold_index)
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"teacher_checkpoint_template 对应文件不存在: {checkpoint_path}")
        state["current_checkpoint"] = checkpoint_path
        state["supervised_checkpoint"] = checkpoint_path
        return checkpoint_path, None

    supervised_output_root = fold_workspace / "iterations" / "iter_00_supervised"
    existing_checkpoint = supervised_output_root / f"fold_{fold_index}" / "best.pt"
    if existing_checkpoint.exists():
        checkpoint_path = str(existing_checkpoint.resolve())
        state["current_checkpoint"] = checkpoint_path
        state["supervised_checkpoint"] = checkpoint_path
        return checkpoint_path, None

    supervised_config = clone_config(config)
    supervised_config["experiment"]["output_root"] = str(supervised_output_root)
    train_result = train_one_fold(supervised_config, fold_index=fold_index)
    checkpoint_path = str(train_result["checkpoint_path"])
    state["current_checkpoint"] = checkpoint_path
    state["supervised_checkpoint"] = checkpoint_path
    return checkpoint_path, train_result


def run_one_fold(config: dict[str, Any], fold_index: int) -> dict[str, Any]:
    semi_cfg = config["semi_supervised"]
    data_cfg = config["data"]
    pseudo_cfg = semi_cfg.get("pseudo", {})
    active_cfg = semi_cfg.get("active_learning", {})
    retrain_cfg = semi_cfg.get("retrain", {})

    fold_workspace = ensure_dir(Path(semi_cfg["workspace_root"]) / f"fold_{fold_index}")
    state_path = fold_workspace / "state.json"
    pseudo_pool_root = fold_workspace / "pseudo_pool"
    query_root = fold_workspace / "query_pool"
    iterations_root = ensure_dir(fold_workspace / "iterations")

    state = load_state(state_path)
    current_checkpoint, supervised_train_result = resolve_initial_checkpoint(
        config=config,
        fold_index=fold_index,
        fold_workspace=fold_workspace,
        state=state,
    )
    if supervised_train_result is not None:
        state["supervised_train_result"] = supervised_train_result
        save_state(state_path, state)

    base_labeled_stems = get_base_labeled_stems(data_cfg["prepared_root"])
    reviewed_labeled_roots = [str(path) for path in semi_cfg.get("reviewed_labeled_roots", [])]
    previous_reviewed_stems = set(state.get("reviewed_source_stems", []))
    reviewed_stems = get_reviewed_labeled_stems(reviewed_labeled_roots) if reviewed_labeled_roots else set()
    state["reviewed_source_stems"] = sorted(reviewed_stems)

    pseudo_pool_items = filter_items_by_stems(load_pseudo_pool(pseudo_pool_root), reviewed_stems)
    pending_queries = filter_query_pending_against_reviewed(load_query_pending(query_root), reviewed_stems)
    save_query_pending(query_root, pending_queries)

    iteration_summaries = list(state.get("iteration_summaries", []))
    reserve_query_samples = bool(semi_cfg.get("reserve_query_samples", True))
    keep_query_pending = bool(semi_cfg.get("keep_query_pending", True))
    iterations = int(semi_cfg.get("iterations", 2))
    stop_if_no_new_pseudo = bool(semi_cfg.get("stop_if_no_new_pseudo", True))

    for iteration_index in range(int(state.get("completed_iterations", 0)) + 1, iterations + 1):
        excluded_stems = set(base_labeled_stems) | set(reviewed_stems)
        excluded_stems.update(
            {str(item.get("source_stem", item.get("stem"))) for item in pseudo_pool_items}
        )
        if reserve_query_samples:
            excluded_stems.update(
                {str(item.get("source_stem", item.get("stem"))) for item in pending_queries}
            )

        unlabeled_items = build_unlabeled_items(
            images_dir=semi_cfg["unlabeled_images_dir"],
            excluded_stems=excluded_stems,
        )

        if not unlabeled_items:
            break

        iteration_predict_root = iterations_root / f"iter_{iteration_index:02d}_predict"
        predictions = infer_unlabeled_items(
            checkpoint_path=current_checkpoint,
            image_items=unlabeled_items,
            output_dir=iteration_predict_root,
            device_name=str(semi_cfg.get("device", config["training"].get("device", "auto"))),
            binary_threshold=float(pseudo_cfg.get("binary_threshold", config["training"].get("threshold", 0.5))),
            use_tta=bool(pseudo_cfg.get("use_tta", True)),
        )

        pseudo_selected = select_pseudo_candidates(
            predictions=predictions,
            confidence_threshold=float(pseudo_cfg.get("confidence_threshold", 0.85)),
            max_items=int(pseudo_cfg.get("max_items_per_iteration", 0)) or None,
            score_name=str(pseudo_cfg.get("score_name", "confidence_score")),
        )
        pseudo_selected_stems = {str(item["stem"]) for item in pseudo_selected}
        active_candidates = [
            item for item in predictions if str(item.get("stem", "")) not in pseudo_selected_stems
        ]
        active_selected = select_active_learning_candidates(
            predictions=active_candidates,
            top_k=int(active_cfg.get("top_k", 5)),
            score_name=str(active_cfg.get("score_name", "uncertainty_score")),
        )

        save_csv(iteration_predict_root / "pseudo_selected.csv", pseudo_selected)
        save_stem_list(
            iteration_predict_root / "pseudo_selected.txt",
            [str(item["stem"]) for item in pseudo_selected],
        )
        export_active_learning_selection(query_root=query_root, selected_items=active_selected, iteration_index=iteration_index)

        if keep_query_pending:
            existing_query_stems = {str(item.get("source_stem", item.get("stem"))) for item in pending_queries}
            for item in active_selected:
                source_stem = str(item["stem"])
                if source_stem in existing_query_stems:
                    continue
                pending_queries.append(
                    {
                        **item,
                        "source_stem": source_stem,
                        "query_iteration": int(iteration_index),
                    }
                )
                existing_query_stems.add(source_stem)
            save_query_pending(query_root, pending_queries)

        pseudo_pool_items = append_pseudo_pool(
            pseudo_pool_root=pseudo_pool_root,
            selected_items=pseudo_selected,
            selected_iteration=iteration_index,
        )
        pseudo_pool_items = filter_items_by_stems(pseudo_pool_items, reviewed_stems)

        has_new_training_signal = bool(pseudo_selected) or (reviewed_stems != previous_reviewed_stems)
        if not has_new_training_signal and stop_if_no_new_pseudo:
            summary = summarize_iteration(
                iteration_index=iteration_index,
                checkpoint_path=current_checkpoint,
                pseudo_selected=pseudo_selected,
                active_selected=active_selected,
                training_result=None,
                pool_info={
                    "unlabeled_remaining": len(unlabeled_items) - len(pseudo_selected) - len(active_selected),
                    "pseudo_pool_size": len(pseudo_pool_items),
                    "query_pending_size": len(pending_queries),
                },
            )
            iteration_summaries.append(summary)
            state["iteration_summaries"] = iteration_summaries
            state["completed_iterations"] = iteration_index
            state["current_checkpoint"] = current_checkpoint
            save_state(state_path, state)
            break

        prepared_root = iterations_root / f"iter_{iteration_index:02d}_prepared"
        merge_info = build_iteration_training_root(
            base_supervised_root=data_cfg["prepared_root"],
            fold_manifest_name=str(data_cfg.get("fold_manifest_name", "folds_3_seed42.json")),
            fold_index=fold_index,
            reviewed_labeled_roots=reviewed_labeled_roots,
            pseudo_pool_items=pseudo_pool_items,
            output_root=prepared_root,
            real_loss_weight=float(retrain_cfg.get("real_loss_weight", 1.0)),
            pseudo_loss_weight=float(retrain_cfg.get("pseudo_loss_weight", 0.35)),
            supervised_repeat=int(retrain_cfg.get("supervised_repeat", 8)),
        )

        train_config = build_iteration_train_config(
            base_config=config,
            prepared_root=merge_info["prepared_root"],
            output_root=str(iterations_root / f"iter_{iteration_index:02d}_train"),
            teacher_checkpoint=current_checkpoint,
            retrain_cfg=retrain_cfg,
            iteration_index=iteration_index,
            teacher_fold_index=fold_index,
        )
        teacher_checkpoint_for_iteration = current_checkpoint
        train_result = train_one_fold(train_config, fold_index=0)
        current_checkpoint = str(train_result["checkpoint_path"])

        summary = summarize_iteration(
            iteration_index=iteration_index,
            checkpoint_path=teacher_checkpoint_for_iteration,
            pseudo_selected=pseudo_selected,
            active_selected=active_selected,
            training_result={
                **train_result,
                "merge_info": merge_info,
                "student_checkpoint": current_checkpoint,
            },
            pool_info={
                "unlabeled_remaining": len(unlabeled_items) - len(pseudo_selected) - len(active_selected),
                "pseudo_pool_size": len(pseudo_pool_items),
                "query_pending_size": len(pending_queries),
            },
        )
        iteration_summaries.append(summary)
        state["iteration_summaries"] = iteration_summaries
        state["completed_iterations"] = iteration_index
        state["current_checkpoint"] = current_checkpoint
        previous_reviewed_stems = set(reviewed_stems)
        save_state(state_path, state)

    summary = {
        "fold_index": int(fold_index),
        "supervised_checkpoint": str(state.get("supervised_checkpoint", "")),
        "current_checkpoint": str(state.get("current_checkpoint", "")),
        "completed_iterations": int(state.get("completed_iterations", 0)),
        "pseudo_pool_size": len(filter_items_by_stems(load_pseudo_pool(pseudo_pool_root), reviewed_stems)),
        "query_pending_size": len(filter_query_pending_against_reviewed(load_query_pending(query_root), reviewed_stems)),
        "iteration_summaries": list(state.get("iteration_summaries", [])),
    }
    save_json(fold_workspace / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if "semi_supervised" not in config:
        raise KeyError("当前配置缺少 semi_supervised 段，无法执行半监督迭代")

    seed_everything(int(config["experiment"].get("seed", 42)))
    maybe_prepare_data(config)

    workspace_root = ensure_dir(config["semi_supervised"]["workspace_root"])
    fold_summaries = [run_one_fold(config=config, fold_index=fold_index) for fold_index in resolve_fold_indices(config, args)]
    final_summary = {
        "experiment": str(config["experiment"]["name"]),
        "workspace_root": str(workspace_root),
        "fold_summaries": fold_summaries,
    }
    save_json(workspace_root / "summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
