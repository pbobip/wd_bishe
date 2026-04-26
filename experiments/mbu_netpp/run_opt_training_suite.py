from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.mbu_netpp.common import ensure_dir, load_yaml, save_json


BASE_CONFIGS = [
    "opt_real53_baseline.yaml",
    "opt_real53_edge_up.yaml",
    "opt_real53_deep_up.yaml",
    "opt_real53_edge_deep_up.yaml",
    "opt_real53_vf_002.yaml",
    "opt_real53_vf_005.yaml",
    "opt_real53_boundary_sampling.yaml",
    "opt_real53_boundary_hard_sampling.yaml",
    "opt_real53_nasa_train_only.yaml",
    "opt_real53_merged78_cv5_ext10.yaml",
]


def build_experiment_tasks(experiment_name: str, num_folds: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    fold_task_ids = []
    for fold_index in range(num_folds):
        task_id = f"{experiment_name}:train_fold:{fold_index}"
        tasks.append(
            {
                "id": task_id,
                "kind": "train_fold",
                "experiment": experiment_name,
                "fold_index": fold_index,
                "deps": [],
                "status": "pending",
                "retries": 0,
            }
        )
        fold_task_ids.append(task_id)
    summarize_id = f"{experiment_name}:summarize"
    tasks.append(
        {
            "id": summarize_id,
            "kind": "summarize",
            "experiment": experiment_name,
            "deps": list(fold_task_ids),
            "status": "pending",
            "retries": 0,
        }
    )
    tasks.append(
        {
            "id": f"{experiment_name}:plot",
            "kind": "plot",
            "experiment": experiment_name,
            "deps": [summarize_id],
            "status": "pending",
            "retries": 0,
        }
    )
    tasks.append(
        {
            "id": f"{experiment_name}:holdout",
            "kind": "holdout",
            "experiment": experiment_name,
            "deps": [summarize_id],
            "status": "pending",
            "retries": 0,
        }
    )
    return tasks


def resolve_default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行优化实验可恢复训练套件")
    parser.add_argument("--repo-root", default=str(resolve_default_repo_root()))
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--num-test", type=int, default=10)
    parser.add_argument("--real-train-source", default="")
    parser.add_argument("--aux-prepared-root", default="")
    parser.add_argument("--micronet-checkpoint", default="")
    parser.add_argument("--delivery-name", default="opt_real53_suite")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    repo_root = Path(args.repo_root).resolve()
    server_assets_root = repo_root / "server_assets"
    real_train_source = (
        Path(args.real_train_source).resolve() if args.real_train_source else (server_assets_root / "training_sources" / "real53")
    )
    aux_prepared_root = (
        Path(args.aux_prepared_root).resolve()
        if args.aux_prepared_root
        else (server_assets_root / "prepared_sources" / "nasa_secondary")
    )
    micronet_checkpoint = (
        Path(args.micronet_checkpoint).resolve()
        if args.micronet_checkpoint
        else (repo_root / "experiments" / "mbu_netpp" / "workdir" / "weights" / "se_resnext50_32x4d_pretrained_microscopynet_v1.0.pth.tar")
    )
    suite_output_root = repo_root / "experiments" / "mbu_netpp" / "outputs" / args.delivery_name
    suite_experiments_root = suite_output_root / "experiments"
    state_root = suite_output_root / "_suite_state"
    delivery_root = repo_root / "results" / "server_delivery"
    delivery_dir = delivery_root / args.delivery_name
    return {
        "repo_root": repo_root,
        "server_assets_root": server_assets_root,
        "real_train_source": real_train_source,
        "aux_prepared_root": aux_prepared_root,
        "micronet_checkpoint": micronet_checkpoint,
        "real_prepared_root": repo_root / "experiments" / "mbu_netpp" / "workdir" / "prepared_real53_cv5",
        "nasa_train_only_root": repo_root / "experiments" / "mbu_netpp" / "workdir" / "prepared_real53_nasa_train_only",
        "merged78_root": repo_root / "experiments" / "mbu_netpp" / "workdir" / "prepared_real53_merged78_cv5_ext10",
        "runtime_config_dir": repo_root / "experiments" / "mbu_netpp" / "workdir" / "runtime_configs_opt_suite",
        "suite_output_root": suite_output_root,
        "suite_experiments_root": suite_experiments_root,
        "traditional_output_root": suite_output_root / "traditional_eval",
        "suite_summary_root": suite_output_root / "suite_summary",
        "state_root": state_root,
        "heartbeat_root": state_root / "heartbeats",
        "task_log_root": state_root / "task_logs",
        "runtime_records_csv": delivery_root / f"{args.delivery_name}_runtime_records.csv",
        "environment_report": delivery_root / f"{args.delivery_name}_environment_report.json",
        "delivery_root": delivery_root,
        "delivery_dir": delivery_dir,
        "delivery_archive": delivery_root / f"{args.delivery_name}.tar.gz",
    }


def write_environment_report(path: Path, python_bin: str) -> None:
    payload = {
        "python_executable": python_bin,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        payload["gpu_name"] = props.name
        payload["gpu_total_gb"] = round(props.total_memory / (1024**3), 2)
    save_json(path, payload)


def append_runtime_record(path: Path, label: str, seconds: float, return_code: int) -> None:
    ensure_dir(path.parent)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(["label", "seconds", "return_code"])
        writer.writerow([label, f"{seconds:.2f}", return_code])


def run_sync(label: str, command: list[str], cwd: Path, runtime_records_csv: Path) -> None:
    start = time.time()
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    append_runtime_record(runtime_records_csv, label, time.time() - start, completed.returncode)
    if completed.returncode != 0:
        raise RuntimeError(f"步骤失败: {label}")


def ensure_real_dataset(paths: dict[str, Path], python_bin: str, num_folds: int, num_test: int, seed: int) -> None:
    dataset_manifest = paths["real_prepared_root"] / "manifests" / "dataset.json"
    holdout_manifest = paths["real_prepared_root"] / "manifests" / f"holdout_{num_test}_seed{seed}.json"
    fold_manifest = paths["real_prepared_root"] / "manifests" / f"folds_{num_folds}_seed{seed}_holdout{num_test}.json"
    if not dataset_manifest.exists():
        run_sync(
            "prepare:real53",
            [
                python_bin,
                "-m",
                "experiments.mbu_netpp.preparation",
                "--images-dir",
                str(paths["real_train_source"]),
                "--annotations-dir",
                str(paths["real_train_source"]),
                "--output-root",
                str(paths["real_prepared_root"]),
                "--labels",
                "gamma_prime",
                "--num-folds",
                str(num_folds),
                "--seed",
                str(seed),
            ],
            cwd=paths["repo_root"],
            runtime_records_csv=paths["runtime_records_csv"],
        )
    if not holdout_manifest.exists() or not fold_manifest.exists():
        run_sync(
            "split:holdout10",
            [
                python_bin,
                "-m",
                "experiments.mbu_netpp.create_holdout_split",
                "--prepared-root",
                str(paths["real_prepared_root"]),
                "--num-test",
                str(num_test),
                "--num-folds",
                str(num_folds),
                "--seed",
                str(seed),
                "--preferred-test-stem",
                "4233",
                "--preferred-test-stem",
                "5",
                "--preferred-test-stem",
                "732",
                "--preferred-test-stem",
                "7323",
            ],
            cwd=paths["repo_root"],
            runtime_records_csv=paths["runtime_records_csv"],
        )


def ensure_traditional_eval(paths: dict[str, Path], python_bin: str, num_folds: int, num_test: int, seed: int) -> None:
    summary_csv = paths["traditional_output_root"] / "traditional_summary.csv"
    if summary_csv.exists():
        return
    run_sync(
        "eval:traditional",
        [
            python_bin,
            "-m",
            "experiments.mbu_netpp.traditional_eval",
            "--prepared-root",
            str(paths["real_prepared_root"]),
            "--fold-manifest-name",
            f"folds_{num_folds}_seed{seed}_holdout{num_test}.json",
            "--fold-index",
            "0",
            "--output-dir",
            str(paths["traditional_output_root"]),
        ],
        cwd=paths["repo_root"],
        runtime_records_csv=paths["runtime_records_csv"],
    )


def ensure_opt_datasets(paths: dict[str, Path], python_bin: str, num_folds: int, num_test: int, seed: int) -> None:
    holdout_manifest_name = f"holdout_{num_test}_seed{seed}.json"
    fold_manifest_name = f"folds_{num_folds}_seed{seed}_holdout{num_test}.json"

    if not (paths["nasa_train_only_root"] / "manifests" / "dataset.json").exists():
        run_sync(
            "prepare:nasa_train_only",
            [
                python_bin,
                "-m",
                "experiments.mbu_netpp.prepare_opt_training_sets",
                "--real-prepared-root",
                str(paths["real_prepared_root"]),
                "--aux-prepared-root",
                str(paths["aux_prepared_root"]),
                "--output-root",
                str(paths["nasa_train_only_root"]),
                "--mode",
                "train_only_aux",
                "--num-folds",
                str(num_folds),
                "--seed",
                str(seed),
                "--holdout-manifest-name",
                holdout_manifest_name,
                "--real-fold-manifest-name",
                fold_manifest_name,
                "--aux-sample-weight",
                "0.25",
                "--aux-sampling-weight",
                "0.15",
            ],
            cwd=paths["repo_root"],
            runtime_records_csv=paths["runtime_records_csv"],
        )

    if not (paths["merged78_root"] / "manifests" / "dataset.json").exists():
        run_sync(
            "prepare:merged78_ext10",
            [
                python_bin,
                "-m",
                "experiments.mbu_netpp.prepare_opt_training_sets",
                "--real-prepared-root",
                str(paths["real_prepared_root"]),
                "--aux-prepared-root",
                str(paths["aux_prepared_root"]),
                "--output-root",
                str(paths["merged78_root"]),
                "--mode",
                "merged_trainval_with_external_holdout",
                "--num-folds",
                str(num_folds),
                "--seed",
                str(seed),
                "--holdout-manifest-name",
                holdout_manifest_name,
                "--real-fold-manifest-name",
                fold_manifest_name,
                "--aux-sample-weight",
                "0.5",
                "--aux-sampling-weight",
                "0.25",
            ],
            cwd=paths["repo_root"],
            runtime_records_csv=paths["runtime_records_csv"],
        )


def build_runtime_configs(paths: dict[str, Path], num_folds: int, seed: int) -> dict[str, dict[str, Any]]:
    ensure_dir(paths["runtime_config_dir"])
    entries: dict[str, dict[str, Any]] = {}
    config_dir = paths["repo_root"] / "experiments" / "mbu_netpp" / "configs"
    for file_name in BASE_CONFIGS:
        base_path = config_dir / file_name
        config = load_yaml(base_path)
        experiment_name = str(config["experiment"]["name"])
        auxiliary_mode = str(config["data"].get("auxiliary", {}).get("mode", "none"))
        if auxiliary_mode == "train_only_aux":
            prepared_root = paths["nasa_train_only_root"]
        elif auxiliary_mode == "merged_trainval_with_external_holdout":
            prepared_root = paths["merged78_root"]
        else:
            prepared_root = paths["real_prepared_root"]

        experiment_root = paths["suite_experiments_root"] / experiment_name
        config["experiment"]["output_root"] = str(experiment_root)
        config["data"]["prepared_root"] = str(prepared_root)
        config["data"]["num_folds"] = int(num_folds)
        config["data"]["fold_seed"] = int(seed)
        config["model"]["micronet_checkpoint"] = str(paths["micronet_checkpoint"])
        config["training"]["device"] = "cuda"
        config["training"]["batch_size"] = 6
        config["training"]["num_workers"] = 8

        runtime_path = paths["runtime_config_dir"] / file_name
        runtime_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        entries[experiment_name] = {
            "name": experiment_name,
            "runtime_config": runtime_path,
            "experiment_root": experiment_root,
        }
    return entries


def task_done(task: dict[str, Any], experiment_entries: dict[str, dict[str, Any]], paths: dict[str, Path]) -> bool:
    if task["kind"] == "train_fold":
        experiment_root = experiment_entries[task["experiment"]]["experiment_root"]
        return (experiment_root / f"fold_{task['fold_index']}" / "summary.json").exists()
    if task["kind"] == "summarize":
        experiment_root = experiment_entries[task["experiment"]]["experiment_root"]
        return (experiment_root / "crossval_summary.json").exists()
    if task["kind"] == "plot":
        experiment_root = experiment_entries[task["experiment"]]["experiment_root"]
        return (experiment_root / "plots" / "plots_manifest.json").exists()
    if task["kind"] == "holdout":
        experiment_root = experiment_entries[task["experiment"]]["experiment_root"]
        return (experiment_root / "holdout_eval" / "summary.json").exists()
    if task["kind"] == "suite_summary":
        return (paths["suite_summary_root"] / "suite_summary.csv").exists()
    if task["kind"] == "package":
        return False
    return False


def find_best_checkpoint(summary_path: Path) -> str:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    best = max(payload.get("fold_results", []), key=lambda item: float(item.get("best_metric", 0.0)))
    return str(best["checkpoint_path"])


def launch_task(
    slot_index: int,
    task: dict[str, Any],
    experiment_entries: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    python_bin: str,
    num_folds: int,
    num_test: int,
    seed: int,
) -> dict[str, Any]:
    ensure_dir(paths["task_log_root"])
    log_path = paths["task_log_root"] / f"{task['id'].replace(':', '__')}.log"
    log_handle = log_path.open("a", encoding="utf-8")

    if task["kind"] == "train_fold":
        runtime_config = experiment_entries[task["experiment"]]["runtime_config"]
        command = [
            python_bin,
            "-m",
            "experiments.mbu_netpp.train",
            "--config",
            str(runtime_config),
            "--fold",
            str(task["fold_index"]),
            "--resume",
        ]
    elif task["kind"] == "summarize":
        runtime_config = experiment_entries[task["experiment"]]["runtime_config"]
        command = [
            python_bin,
            "-m",
            "experiments.mbu_netpp.train",
            "--config",
            str(runtime_config),
            "--run-all-folds",
            "--summarize-only",
        ]
    elif task["kind"] == "plot":
        command = [
            python_bin,
            "-m",
            "experiments.mbu_netpp.plot_training_curves",
            "--experiment-root",
            str(experiment_entries[task["experiment"]]["experiment_root"]),
        ]
    elif task["kind"] == "holdout":
        experiment_root = experiment_entries[task["experiment"]]["experiment_root"]
        runtime_config = experiment_entries[task["experiment"]]["runtime_config"]
        checkpoint = find_best_checkpoint(experiment_root / "crossval_summary.json")
        command = [
            python_bin,
            "-m",
            "experiments.mbu_netpp.holdout_eval",
            "--checkpoint",
            checkpoint,
            "--prepared-root",
            str(load_yaml(runtime_config)["data"]["prepared_root"]),
            "--fold-manifest-name",
            f"folds_{num_folds}_seed{seed}_holdout{num_test}.json",
            "--fold-index",
            "0",
            "--output-dir",
            str(experiment_root / "holdout_eval"),
            "--config",
            str(runtime_config),
            "--device",
            "auto",
        ]
    elif task["kind"] == "suite_summary":
        command = [
            python_bin,
            "-m",
            "experiments.mbu_netpp.summarize_experiment_suite",
            "--experiments-root",
            str(paths["suite_experiments_root"]),
            "--output-dir",
            str(paths["suite_summary_root"]),
        ]
    elif task["kind"] == "package":
        command = [
            python_bin,
            "-m",
            "experiments.mbu_netpp.run_opt_training_suite",
            "--repo-root",
            str(paths["repo_root"]),
            "--python-bin",
            python_bin,
            "--delivery-name",
            paths["delivery_dir"].name,
            "--max-workers",
            "0",
        ]
        raise RuntimeError("package 任务不通过子进程调用")
    else:
        raise ValueError(f"未知 task kind: {task['kind']}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=str(paths["repo_root"]),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    task["status"] = "running"
    task["started_at"] = time.time()
    return {
        "slot_index": slot_index,
        "task": task,
        "process": process,
        "log_handle": log_handle,
        "log_path": log_path,
    }


def write_heartbeat(paths: dict[str, Path], slot_index: int, active_meta: dict[str, Any]) -> None:
    ensure_dir(paths["heartbeat_root"])
    save_json(
        paths["heartbeat_root"] / f"worker_{slot_index}.json",
        {
            "slot_index": slot_index,
            "task_id": active_meta["task"]["id"],
            "status": active_meta["task"]["status"],
            "pid": active_meta["process"].pid,
            "updated_at": time.time(),
        },
    )


def write_state_snapshot(paths: dict[str, Path], tasks: list[dict[str, Any]], active: dict[int, dict[str, Any]]) -> None:
    payload = {
        "updated_at": time.time(),
        "counts": {
            "pending": sum(1 for task in tasks if task["status"] == "pending"),
            "running": sum(1 for task in tasks if task["status"] == "running"),
            "done": sum(1 for task in tasks if task["status"] == "done"),
            "failed": sum(1 for task in tasks if task["status"] == "failed"),
        },
        "tasks": tasks,
        "active_slots": {
            str(slot_index): {
                "task_id": meta["task"]["id"],
                "pid": meta["process"].pid,
                "started_at": meta["task"].get("started_at"),
                "log_path": str(meta["log_path"]),
            }
            for slot_index, meta in active.items()
        },
    }
    save_json(paths["state_root"] / "tasks.json", payload)


def package_results(paths: dict[str, Path], experiment_entries: dict[str, dict[str, Any]]) -> None:
    if paths["delivery_dir"].exists():
        shutil.rmtree(paths["delivery_dir"])
    ensure_dir(paths["delivery_dir"])
    ensure_dir(paths["delivery_dir"] / "manifests" / "real53")
    ensure_dir(paths["delivery_dir"] / "manifests" / "nasa_train_only")
    ensure_dir(paths["delivery_dir"] / "manifests" / "merged78_cv5_ext10")
    shutil.copytree(paths["traditional_output_root"], paths["delivery_dir"] / "traditional_eval")
    shutil.copytree(paths["suite_summary_root"], paths["delivery_dir"] / "suite_summary")

    for manifest_root, label in [
        (paths["real_prepared_root"], "real53"),
        (paths["nasa_train_only_root"], "nasa_train_only"),
        (paths["merged78_root"], "merged78_cv5_ext10"),
    ]:
        source_dir = manifest_root / "manifests"
        target_dir = paths["delivery_dir"] / "manifests" / label
        for file in source_dir.glob("*.json"):
            shutil.copy2(file, target_dir / file.name)

    experiments_target = paths["delivery_dir"] / "experiments"
    ensure_dir(experiments_target)
    for experiment_name, entry in experiment_entries.items():
        experiment_root = entry["experiment_root"]
        target_root = experiments_target / experiment_name
        ensure_dir(target_root)
        shutil.copy2(entry["runtime_config"], target_root / "runtime_config.yaml")
        shutil.copy2(experiment_root / "crossval_summary.json", target_root / "crossval_summary.json")
        shutil.copytree(experiment_root / "plots", target_root / "plots")
        shutil.copytree(experiment_root / "holdout_eval", target_root / "holdout_eval")
        shutil.copy2(find_best_checkpoint(experiment_root / "crossval_summary.json"), target_root / "best.pt")
        for fold_dir in experiment_root.glob("fold_*"):
            fold_target = target_root / fold_dir.name
            ensure_dir(fold_target)
            for name in ["summary.json", "history.json"]:
                source = fold_dir / name
                if source.exists():
                    shutil.copy2(source, fold_target / name)

    shutil.copy2(paths["environment_report"], paths["delivery_dir"] / "environment_report.json")
    if paths["runtime_records_csv"].exists():
        shutil.copy2(paths["runtime_records_csv"], paths["delivery_dir"] / "runtime_records.csv")
    run_summary = {
        "delivery_dir": str(paths["delivery_dir"]),
        "delivery_archive": str(paths["delivery_archive"]),
        "suite_output_root": str(paths["suite_output_root"]),
        "suite_experiments_root": str(paths["suite_experiments_root"]),
        "traditional_output_root": str(paths["traditional_output_root"]),
        "suite_summary_root": str(paths["suite_summary_root"]),
    }
    save_json(paths["delivery_dir"] / "run_summary.json", run_summary)
    if paths["delivery_archive"].exists():
        paths["delivery_archive"].unlink()
    with tarfile.open(paths["delivery_archive"], "w:gz") as handle:
        handle.add(paths["delivery_dir"], arcname=paths["delivery_dir"].name)


def refresh_done_status(tasks: list[dict[str, Any]], experiment_entries: dict[str, dict[str, Any]], paths: dict[str, Path]) -> None:
    for task in tasks:
        if task["status"] == "running":
            continue
        if task_done(task, experiment_entries, paths):
            task["status"] = "done"


def deps_satisfied(task: dict[str, Any], tasks_by_id: dict[str, dict[str, Any]]) -> bool:
    return all(tasks_by_id[task_id]["status"] == "done" for task_id in task.get("deps", []))


def main() -> None:
    args = parse_args()
    if not args.python_bin:
        args.python_bin = sys.executable

    paths = resolve_paths(args)
    ensure_dir(paths["delivery_root"])
    ensure_dir(paths["suite_output_root"])
    ensure_dir(paths["suite_experiments_root"])
    ensure_dir(paths["state_root"])

    write_environment_report(paths["environment_report"], args.python_bin)
    ensure_real_dataset(paths, args.python_bin, args.num_folds, args.num_test, args.seed)
    ensure_traditional_eval(paths, args.python_bin, args.num_folds, args.num_test, args.seed)
    ensure_opt_datasets(paths, args.python_bin, args.num_folds, args.num_test, args.seed)
    experiment_entries = build_runtime_configs(paths, args.num_folds, args.seed)

    tasks: list[dict[str, Any]] = []
    for experiment_name in experiment_entries:
        tasks.extend(build_experiment_tasks(experiment_name, args.num_folds))
    tasks.append(
        {
            "id": "suite_summary",
            "kind": "suite_summary",
            "deps": [
                dependency
                for experiment_name in experiment_entries
                for dependency in (f"{experiment_name}:holdout", f"{experiment_name}:plot")
            ],
            "status": "pending",
            "retries": 0,
        }
    )
    tasks.append(
        {
            "id": "package",
            "kind": "package",
            "deps": [
                "suite_summary",
                *[
                    dependency
                    for experiment_name in experiment_entries
                    for dependency in (f"{experiment_name}:holdout", f"{experiment_name}:plot")
                ],
            ],
            "status": "pending",
            "retries": 0,
        }
    )
    tasks_by_id = {task["id"]: task for task in tasks}
    refresh_done_status(tasks, experiment_entries, paths)

    active: dict[int, dict[str, Any]] = {}
    while True:
        refresh_done_status(tasks, experiment_entries, paths)

        finished_slots: list[int] = []
        for slot_index, meta in active.items():
            process = meta["process"]
            write_heartbeat(paths, slot_index, meta)
            return_code = process.poll()
            if return_code is None:
                continue
            meta["log_handle"].close()
            append_runtime_record(
                paths["runtime_records_csv"],
                meta["task"]["id"],
                time.time() - meta["task"].get("started_at", time.time()),
                return_code,
            )
            if return_code == 0 and task_done(meta["task"], experiment_entries, paths):
                meta["task"]["status"] = "done"
            else:
                meta["task"]["retries"] += 1
                if meta["task"]["retries"] <= args.max_retries:
                    meta["task"]["status"] = "pending"
                else:
                    meta["task"]["status"] = "failed"
            finished_slots.append(slot_index)
        for slot_index in finished_slots:
            active.pop(slot_index, None)

        if all(task["status"] == "done" for task in tasks):
            break
        if any(task["status"] == "failed" for task in tasks) and not active:
            write_state_snapshot(paths, tasks, active)
            raise RuntimeError("存在失败任务且已达到最大重试次数")

        pending_tasks = [
            task
            for task in tasks
            if task["status"] == "pending" and deps_satisfied(task, tasks_by_id)
        ]
        free_slots = [index for index in range(max(args.max_workers, 0)) if index not in active]
        for slot_index in free_slots:
            if not pending_tasks:
                break
            task = pending_tasks.pop(0)
            if task["kind"] == "package":
                start = time.time()
                package_results(paths, experiment_entries)
                append_runtime_record(paths["runtime_records_csv"], "package", time.time() - start, 0)
                task["status"] = "done"
                continue
            active[slot_index] = launch_task(
                slot_index=slot_index,
                task=task,
                experiment_entries=experiment_entries,
                paths=paths,
                python_bin=args.python_bin,
                num_folds=args.num_folds,
                num_test=args.num_test,
                seed=args.seed,
            )

        write_state_snapshot(paths, tasks, active)
        time.sleep(max(args.poll_seconds, 1))

    write_state_snapshot(paths, tasks, active)


if __name__ == "__main__":
    main()
