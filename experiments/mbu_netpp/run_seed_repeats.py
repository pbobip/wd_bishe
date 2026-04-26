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


METRIC_KEYS = ["dice", "iou", "precision", "recall", "vf", "boundary_f1"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_base_config() -> Path:
    return repo_root() / "experiments" / "mbu_netpp" / "configs" / "opt_real53_boundary_sampling.yaml"


def default_output_root() -> Path:
    return repo_root() / "experiments" / "mbu_netpp" / "outputs" / "paper_seed_repeats"


def default_delivery_root() -> Path:
    return repo_root() / "results" / "server_delivery"


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def append_runtime_record(path: Path, label: str, seconds: float, return_code: int) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["label", "seconds", "return_code"])
        writer.writerow([label, f"{seconds:.2f}", return_code])


def write_environment_report(path: Path, python_bin: str) -> None:
    payload: dict[str, Any] = {
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return float(variance**0.5)


def build_seed_configs(args: argparse.Namespace, root: Path) -> dict[int, dict[str, Any]]:
    base_config_path = resolve_path(args.base_config, root)
    base_config = load_yaml(base_config_path)
    runtime_config_dir = ensure_dir(resolve_path(args.runtime_config_dir, root))
    output_root = ensure_dir(resolve_path(args.output_root, root))
    prepared_root = resolve_path(args.prepared_root or base_config["data"]["prepared_root"], root)
    micronet_checkpoint = resolve_path(args.micronet_checkpoint or base_config["model"]["micronet_checkpoint"], root)

    entries: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        config = json.loads(json.dumps(base_config))
        experiment_name = f"{base_config['experiment']['name']}_seed{seed}"
        experiment_root = output_root / "experiments" / experiment_name
        config["experiment"]["name"] = experiment_name
        config["experiment"]["seed"] = int(seed)
        config["experiment"]["output_root"] = str(experiment_root)
        config["data"]["prepared_root"] = str(prepared_root)
        config["data"]["fold_seed"] = int(args.fold_seed)
        config["data"]["fold_manifest_name"] = str(args.fold_manifest_name or base_config["data"]["fold_manifest_name"])
        config["data"]["holdout_manifest_name"] = str(
            args.holdout_manifest_name or base_config["data"].get("holdout_manifest_name", "")
        )
        config["model"]["micronet_checkpoint"] = str(micronet_checkpoint)
        config["training"]["device"] = str(args.device)
        config["training"]["batch_size"] = int(args.batch_size)
        config["training"]["num_workers"] = int(args.num_workers)

        runtime_config = runtime_config_dir / f"{experiment_name}.yaml"
        ensure_dir(runtime_config.parent)
        runtime_config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        entries[int(seed)] = {
            "seed": int(seed),
            "experiment_name": experiment_name,
            "experiment_root": experiment_root,
            "runtime_config": runtime_config,
            "config": config,
        }
    return entries


def build_tasks(seed_entries: dict[int, dict[str, Any]], num_folds: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for seed, entry in seed_entries.items():
        train_deps: list[str] = []
        for fold_index in range(num_folds):
            task_id = f"seed{seed}:fold{fold_index}"
            tasks.append(
                {
                    "id": task_id,
                    "kind": "train_fold",
                    "seed": seed,
                    "fold_index": fold_index,
                    "deps": [],
                    "status": "pending",
                    "retries": 0,
                }
            )
            train_deps.append(task_id)
        summarize_id = f"seed{seed}:summarize"
        tasks.append(
            {
                "id": summarize_id,
                "kind": "summarize",
                "seed": seed,
                "deps": train_deps,
                "status": "pending",
                "retries": 0,
            }
        )
        tasks.append(
            {
                "id": f"seed{seed}:plot",
                "kind": "plot",
                "seed": seed,
                "deps": [summarize_id],
                "status": "pending",
                "retries": 0,
            }
        )
        tasks.append(
            {
                "id": f"seed{seed}:holdout",
                "kind": "holdout",
                "seed": seed,
                "deps": [summarize_id],
                "status": "pending",
                "retries": 0,
            }
        )

    holdout_deps = [f"seed{seed}:holdout" for seed in seed_entries]
    plot_deps = [f"seed{seed}:plot" for seed in seed_entries]
    tasks.append(
        {
            "id": "aggregate",
            "kind": "aggregate",
            "deps": holdout_deps + plot_deps,
            "status": "pending",
            "retries": 0,
        }
    )
    tasks.append(
        {
            "id": "package",
            "kind": "package",
            "deps": ["aggregate"],
            "status": "pending",
            "retries": 0,
        }
    )
    return tasks


def find_best_checkpoint(crossval_summary_path: Path) -> Path:
    payload = load_json(crossval_summary_path)
    best = max(payload.get("fold_results", []), key=lambda item: float(item.get("best_metric", 0.0)))
    return Path(best["checkpoint_path"])


def task_done(task: dict[str, Any], seed_entries: dict[int, dict[str, Any]], args: argparse.Namespace) -> bool:
    if task["kind"] == "train_fold":
        root = seed_entries[int(task["seed"])]["experiment_root"]
        return (root / f"fold_{task['fold_index']}" / "summary.json").exists()
    if task["kind"] == "summarize":
        root = seed_entries[int(task["seed"])]["experiment_root"]
        return (root / "crossval_summary.json").exists()
    if task["kind"] == "plot":
        root = seed_entries[int(task["seed"])]["experiment_root"]
        return (root / "plots" / "plots_manifest.json").exists()
    if task["kind"] == "holdout":
        root = seed_entries[int(task["seed"])]["experiment_root"]
        return (root / "holdout_eval" / "summary.json").exists()
    if task["kind"] == "aggregate":
        return (resolve_path(args.output_root, repo_root()) / "seed_repeat_summary" / "seed_repeat_summary.json").exists()
    if task["kind"] == "package":
        return False
    return False


def command_for_task(
    task: dict[str, Any],
    seed_entries: dict[int, dict[str, Any]],
    python_bin: str,
    args: argparse.Namespace,
) -> list[str]:
    if task["kind"] == "train_fold":
        entry = seed_entries[int(task["seed"])]
        return [
            python_bin,
            "-m",
            "experiments.mbu_netpp.train",
            "--config",
            str(entry["runtime_config"]),
            "--fold",
            str(task["fold_index"]),
            "--resume",
        ]
    if task["kind"] == "summarize":
        entry = seed_entries[int(task["seed"])]
        return [
            python_bin,
            "-m",
            "experiments.mbu_netpp.train",
            "--config",
            str(entry["runtime_config"]),
            "--run-all-folds",
            "--summarize-only",
        ]
    if task["kind"] == "plot":
        entry = seed_entries[int(task["seed"])]
        return [
            python_bin,
            "-m",
            "experiments.mbu_netpp.plot_training_curves",
            "--experiment-root",
            str(entry["experiment_root"]),
        ]
    if task["kind"] == "holdout":
        entry = seed_entries[int(task["seed"])]
        config = entry["config"]
        checkpoint = find_best_checkpoint(entry["experiment_root"] / "crossval_summary.json")
        return [
            python_bin,
            "-m",
            "experiments.mbu_netpp.holdout_eval",
            "--checkpoint",
            str(checkpoint),
            "--prepared-root",
            str(config["data"]["prepared_root"]),
            "--fold-manifest-name",
            str(config["data"]["fold_manifest_name"]),
            "--fold-index",
            "0",
            "--output-dir",
            str(entry["experiment_root"] / "holdout_eval"),
            "--config",
            str(entry["runtime_config"]),
            "--device",
            str(args.device),
        ]
    raise ValueError(f"Task {task['kind']} is not a subprocess task")


def deps_satisfied(task: dict[str, Any], task_by_id: dict[str, dict[str, Any]]) -> bool:
    return all(task_by_id[dep]["status"] == "done" for dep in task.get("deps", []))


def refresh_done(tasks: list[dict[str, Any]], seed_entries: dict[int, dict[str, Any]], args: argparse.Namespace) -> None:
    for task in tasks:
        if task["status"] == "running":
            continue
        if task_done(task, seed_entries, args):
            task["status"] = "done"


def write_state(path: Path, tasks: list[dict[str, Any]], active: dict[int, dict[str, Any]]) -> None:
    payload = {
        "updated_at": time.time(),
        "counts": {
            "pending": sum(1 for task in tasks if task["status"] == "pending"),
            "running": sum(1 for task in tasks if task["status"] == "running"),
            "done": sum(1 for task in tasks if task["status"] == "done"),
            "failed": sum(1 for task in tasks if task["status"] == "failed"),
        },
        "active": {
            str(slot): {
                "task_id": meta["task"]["id"],
                "pid": meta["process"].pid,
                "log_path": str(meta["log_path"]),
                "started_at": meta["task"].get("started_at"),
            }
            for slot, meta in active.items()
        },
        "tasks": tasks,
    }
    save_json(path, payload)


def launch_task(
    slot_index: int,
    task: dict[str, Any],
    seed_entries: dict[int, dict[str, Any]],
    python_bin: str,
    args: argparse.Namespace,
    log_root: Path,
    cwd: Path,
) -> dict[str, Any]:
    ensure_dir(log_root)
    log_path = log_root / f"{task['id'].replace(':', '__')}.log"
    log_handle = log_path.open("a", encoding="utf-8")
    command = command_for_task(task, seed_entries, python_bin, args)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(command, cwd=str(cwd), stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    task["status"] = "running"
    task["started_at"] = time.time()
    return {
        "slot_index": slot_index,
        "task": task,
        "process": process,
        "log_handle": log_handle,
        "log_path": log_path,
    }


def aggregate_seed_repeats(seed_entries: dict[int, dict[str, Any]], output_root: Path) -> dict[str, Any]:
    summary_dir = ensure_dir(output_root / "seed_repeat_summary")
    rows: list[dict[str, Any]] = []
    for seed, entry in seed_entries.items():
        crossval = load_json(entry["experiment_root"] / "crossval_summary.json")
        holdout = load_json(entry["experiment_root"] / "holdout_eval" / "summary.json")
        row: dict[str, Any] = {
            "seed": seed,
            "experiment": entry["experiment_name"],
            "crossval_summary_path": str(entry["experiment_root"] / "crossval_summary.json"),
            "holdout_summary_path": str(entry["experiment_root"] / "holdout_eval" / "summary.json"),
        }
        for key in METRIC_KEYS:
            row[f"crossval_{key}"] = float(crossval.get("mean_best_summary", {}).get(key, 0.0))
            row[f"holdout_{key}"] = float(holdout.get("metrics", {}).get(key, 0.0))
        rows.append(row)

    aggregate: dict[str, Any] = {
        "num_seeds": len(rows),
        "seeds": [row["seed"] for row in rows],
        "rows": rows,
        "metric_summary": {},
    }
    for prefix in ["crossval", "holdout"]:
        for key in METRIC_KEYS:
            values = [float(row[f"{prefix}_{key}"]) for row in rows]
            aggregate["metric_summary"][f"{prefix}_{key}"] = {
                "mean": mean(values),
                "std": std(values),
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }

    write_csv(summary_dir / "seed_repeat_summary.csv", rows)
    save_json(summary_dir / "seed_repeat_summary.json", aggregate)
    return aggregate


def copytree_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def package_results(
    seed_entries: dict[int, dict[str, Any]],
    output_root: Path,
    delivery_root: Path,
    delivery_name: str,
    environment_report: Path,
    runtime_records: Path,
) -> Path:
    delivery_dir = delivery_root / delivery_name
    archive_path = delivery_root / f"{delivery_name}.tar.gz"
    if delivery_dir.exists():
        shutil.rmtree(delivery_dir)
    ensure_dir(delivery_dir)
    shutil.copytree(output_root / "seed_repeat_summary", delivery_dir / "seed_repeat_summary")
    if environment_report.exists():
        shutil.copy2(environment_report, delivery_dir / "environment_report.json")
    if runtime_records.exists():
        shutil.copy2(runtime_records, delivery_dir / "runtime_records.csv")

    experiments_dir = ensure_dir(delivery_dir / "experiments")
    for seed, entry in seed_entries.items():
        target = ensure_dir(experiments_dir / entry["experiment_name"])
        shutil.copy2(entry["runtime_config"], target / "runtime_config.yaml")
        shutil.copy2(entry["experiment_root"] / "crossval_summary.json", target / "crossval_summary.json")
        shutil.copy2(find_best_checkpoint(entry["experiment_root"] / "crossval_summary.json"), target / "best.pt")
        copytree_if_exists(entry["experiment_root"] / "plots", target / "plots")
        copytree_if_exists(entry["experiment_root"] / "holdout_eval", target / "holdout_eval")
        for fold_dir in sorted(entry["experiment_root"].glob("fold_*")):
            fold_target = ensure_dir(target / fold_dir.name)
            for name in ["summary.json", "history.json"]:
                source = fold_dir / name
                if source.exists():
                    shutil.copy2(source, fold_target / name)

    save_json(
        delivery_dir / "run_summary.json",
        {
            "delivery_dir": str(delivery_dir),
            "archive_path": str(archive_path),
            "output_root": str(output_root),
            "num_seeds": len(seed_entries),
        },
    )
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as handle:
        handle.add(delivery_dir, arcname=delivery_dir.name)
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final MBU-Net++ seed repeat experiments")
    parser.add_argument("--base-config", default=str(default_base_config()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 2026])
    parser.add_argument("--fold-seed", type=int, default=42, help="Keep this fixed so only training randomness changes")
    parser.add_argument("--fold-manifest-name", default="")
    parser.add_argument("--holdout-manifest-name", default="")
    parser.add_argument("--prepared-root", default="")
    parser.add_argument("--micronet-checkpoint", default="")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--runtime-config-dir", default=str(repo_root() / "experiments" / "mbu_netpp" / "workdir" / "runtime_configs_seed_repeats"))
    parser.add_argument("--delivery-root", default=str(default_delivery_root()))
    parser.add_argument("--delivery-name", default="paper_seed_repeats")
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    python_bin = args.python_bin or sys.executable
    output_root = ensure_dir(resolve_path(args.output_root, root))
    state_root = ensure_dir(output_root / "_state")
    log_root = ensure_dir(state_root / "logs")
    delivery_root = ensure_dir(resolve_path(args.delivery_root, root))
    runtime_records = state_root / "runtime_records.csv"
    environment_report = state_root / "environment_report.json"
    write_environment_report(environment_report, python_bin)

    seed_entries = build_seed_configs(args, root)
    num_folds = int(next(iter(seed_entries.values()))["config"]["data"].get("num_folds", 5))
    tasks = build_tasks(seed_entries, num_folds=num_folds)
    task_by_id = {task["id"]: task for task in tasks}
    refresh_done(tasks, seed_entries, args)
    active: dict[int, dict[str, Any]] = {}

    while True:
        refresh_done(tasks, seed_entries, args)

        finished: list[int] = []
        for slot_index, meta in active.items():
            return_code = meta["process"].poll()
            if return_code is None:
                continue
            meta["log_handle"].close()
            elapsed = time.time() - float(meta["task"].get("started_at", time.time()))
            append_runtime_record(runtime_records, meta["task"]["id"], elapsed, int(return_code))
            if return_code == 0 and task_done(meta["task"], seed_entries, args):
                meta["task"]["status"] = "done"
            else:
                meta["task"]["retries"] += 1
                if meta["task"]["retries"] <= int(args.max_retries):
                    meta["task"]["status"] = "pending"
                else:
                    meta["task"]["status"] = "failed"
            finished.append(slot_index)
        for slot_index in finished:
            active.pop(slot_index, None)

        pending_ready = [
            task
            for task in tasks
            if task["status"] == "pending" and deps_satisfied(task, task_by_id)
        ]
        for task in list(pending_ready):
            if task["kind"] == "aggregate":
                start = time.time()
                aggregate_seed_repeats(seed_entries, output_root)
                append_runtime_record(runtime_records, "aggregate", time.time() - start, 0)
                task["status"] = "done"
                pending_ready.remove(task)
            elif task["kind"] == "package":
                start = time.time()
                archive_path = package_results(
                    seed_entries=seed_entries,
                    output_root=output_root,
                    delivery_root=delivery_root,
                    delivery_name=str(args.delivery_name),
                    environment_report=environment_report,
                    runtime_records=runtime_records,
                )
                append_runtime_record(runtime_records, "package", time.time() - start, 0)
                task["status"] = "done"
                print(f"PACKAGE_ARCHIVE={archive_path}")
                pending_ready.remove(task)

        if all(task["status"] == "done" for task in tasks):
            write_state(state_root / "tasks.json", tasks, active)
            break
        if any(task["status"] == "failed" for task in tasks) and not active:
            write_state(state_root / "tasks.json", tasks, active)
            raise RuntimeError("存在失败任务，且已达到最大重试次数")

        free_slots = [slot for slot in range(max(0, int(args.max_workers))) if slot not in active]
        pending_subprocess = [
            task
            for task in tasks
            if task["status"] == "pending" and deps_satisfied(task, task_by_id) and task["kind"] not in {"aggregate", "package"}
        ]
        for slot_index in free_slots:
            if not pending_subprocess:
                break
            task = pending_subprocess.pop(0)
            active[slot_index] = launch_task(
                slot_index=slot_index,
                task=task,
                seed_entries=seed_entries,
                python_bin=python_bin,
                args=args,
                log_root=log_root,
                cwd=root,
            )

        write_state(state_root / "tasks.json", tasks, active)
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
