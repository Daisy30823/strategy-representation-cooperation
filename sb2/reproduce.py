"""Command-line entry point for the revised submission's reproduction workflows."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from zipfile import ZipFile

CODE = Path(__file__).resolve().parent
REPO = CODE.parent
DATA_SIZE = 535468814
DATA_SHA256 = "cbed752248de4a143da51738008def2d789c7e257475e8e58b93defd62227f0e"
RUNS = ("runA", "runB", "runC", "runD")
LEGACY_DIR = PurePosixPath("v1/submission_package_PNAS_Idea3/06_Dataset_S2_data")
HUMAN_INPUT = LEGACY_DIR / "dbf2019_tableA5_memory1_elicitation.csv"
LEGACY_INPUTS = {LEGACY_DIR / "chains_all.csv", LEGACY_DIR / "numbers.json", HUMAN_INPUT}
EXACT_SPECS = {
    "TAB1": "TAB1:0:0", "PERC1": "PERC1:10:0", "RNN1t": "RNN1t:11:1",
    "AUTO2": "AUTO2:4:2", "CIRC1": "CIRC1:8:2", "PROG1": "PROG1:9:4",
}
FIGURE_NAMES = ("Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "FigS1", "FigS2")


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def script_command(name, *arguments):
    return [sys.executable, str(CODE / name), *map(str, arguments)]


def run_script(name, *arguments, cwd=CODE):
    subprocess.run(script_command(name, *arguments), cwd=cwd, check=True)


def verify_archive(path):
    if path.stat().st_size != DATA_SIZE:
        raise ValueError("Dataset S2 does not match the documented size")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != DATA_SHA256:
        raise ValueError("Dataset S2 does not match the documented SHA-256")


def stage_archive(archive, workspace, quick):
    with ZipFile(archive) as source:
        for item in source.infolist():
            name = PurePosixPath(item.filename)
            archived_result = len(name.parts) == 2 and (
                (name.parts[0] in RUNS and name.name != "tasks.json" and name.suffix in {".json", ".npz"})
                or (name.parts[0] == "exact1" and name.name.startswith("exact1_") and name.suffix == ".json")
            )
            if name not in LEGACY_INPUTS and not (quick and archived_result):
                continue
            if item.is_dir() or ".." in name.parts or "\\" in item.filename:
                raise ValueError(f"Unexpected input member: {item.filename}")
            if name in LEGACY_INPUTS:
                target = workspace / "v1" / name.name
            else:
                target = workspace.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as original, target.open("xb") as copy:
                shutil.copyfileobj(original, copy)


def task_key(task):
    return task["kind"], task["id"]


def run_batch(workspace, name, tasks, workers):
    batch_dir = workspace / name
    batch_dir.mkdir(parents=True, exist_ok=True)
    task_file = batch_dir / "tasks.json"
    task_file.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: {len(tasks)} tasks", flush=True)
    processes = []
    try:
        for index in range(min(workers, len(tasks))):
            command = script_command("run_tasks2.py", task_file, batch_dir, f"worker-{index + 1}")
            processes.append(subprocess.Popen(command, cwd=workspace))
        codes = [process.wait() for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
    if any(codes):
        raise RuntimeError(f"{name} failed. Inspect worker messages and ERR_*.txt in {batch_dir}")

    missing = []
    for task in tasks:
        kind, task_id = task_key(task)
        expected = [f"{kind}_{task_id}.json"]
        if kind in {"tables", "progenum"}:
            expected.append(f"{kind}_{task_id}.npz")
        if kind == "sample2" and task.get("save"):
            expected.append(f"phen_{task_id}.npz")
        missing.extend(filename for filename in expected if not (batch_dir / filename).is_file())
    if missing:
        raise RuntimeError(f"{name} incomplete. Missing outputs include {', '.join(missing[:5])}")


def exact(output, encodings):
    output.mkdir(parents=True, exist_ok=True)
    for encoding in encodings:
        run_script("exact1.py", output, EXACT_SPECS[encoding])


def figures(workspace, output, names):
    from figs2 import make_figures

    make_figures(workspace, output, names)


def tables(workspace, output):
    from tables2 import write_tables

    write_tables(workspace, output)


def priors(workspace):
    run_script("make_mh_priors.py", workspace / "runA", workspace / "priors")


def analyze(workspace, output):
    results = workspace / "res3"
    results.mkdir()
    run_script("human2.py", workspace / "v1" / HUMAN_INPUT.name, results / "human2.json")
    run_script("analysis2.py", workspace)
    figures(workspace, output / "figures", FIGURE_NAMES)
    tables(workspace, output / "tables")
    print(f"Figures and table fragments written to {output}", flush=True)


def full_tasks(workspace, workers):
    tasks = {run: read_json(REPO / "configs" / run / "tasks.json") for run in RUNS}
    revisions = {task_key(task) for task in tasks["runD"]}
    run_b = [task for task in tasks["runB"] if task_key(task) not in revisions]
    run_batch(workspace, "runA", tasks["runA"], workers)
    run_batch(workspace, "runB", run_b, workers)
    run_batch(workspace, "runD", tasks["runD"], workers)
    priors(workspace)
    run_batch(workspace, "runC", tasks["runC"], workers)
    exact(workspace / "exact1", EXACT_SPECS)


def reproduction(archive, output, mode, workers):
    verify_archive(archive)
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    stage_archive(archive, workspace, quick=mode == "quick")
    if mode == "quick":
        run_script("test_core.py")
        priors(workspace)
    else:
        full_tasks(workspace, workers)
    analyze(workspace, output)


def smoke(output):
    import numpy as np

    output.mkdir(parents=True, exist_ok=False)
    base = read_json(REPO / "configs" / "runA" / "tasks.json")[0]
    chain = dict(base, id="short_chain", nsteps=2000, burn=100, rec=100, nwin=10, cap=256, nbhd=False)
    sample = dict(kind="sample2", id="short_sample", enc=0, size=0, M=2000, seed=100001, save=True)
    sample_file = "sample/phen_short_sample.npz"
    mh = dict(chain, id="short_chain_mh", mh={"file": sample_file, "floor": 0.5}, requires=[sample_file])
    snapshots = []
    for trial in ("first", "second"):
        workspace = output / trial
        run_batch(workspace, "sample", [sample], workers=1)
        run_batch(workspace, "chain", [chain, mh], workers=2)
        results = [
            read_json(workspace / "sample/sample2_short_sample.json"),
            read_json(workspace / "chain/chain2_short_chain.json"),
            read_json(workspace / "chain/chain2_short_chain_mh.json"),
        ]
        for result in results:
            result.pop("seconds")
            result.pop("seconds_total", None)
            if result["overflow"] or result["collisions"] or result.get("log_truncated"):
                raise RuntimeError("Smoke test failed its simulation integrity checks")
        with np.load(workspace / sample_file, allow_pickle=False) as arrays:
            if int(arrays["cnt"].sum()) != sample["M"]:
                raise RuntimeError("Smoke test did not account for every sampled genotype")
            snapshots.append((results, dict(arrays)))
    first, second = snapshots
    if first[0] != second[0] or first[1].keys() != second[1].keys() or any(
        not np.array_equal(first[1][key], second[1][key]) for key in first[1]
    ):
        raise RuntimeError("Same-seed smoke outputs differ")
    print("PASS: sampling, baseline chain, relative-path MH chain and same-seed determinism", flush=True)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="mode", required=True)
    commands.add_parser("test", help="run the core model tests")
    for mode in ("smoke", "exact", "quick", "full", "figures", "tables"):
        command = commands.add_parser(mode)
        command.add_argument("--output", type=Path, default=REPO / "outputs" / mode)
        if mode in {"quick", "full"}:
            command.add_argument("--dataset-s2", type=Path, required=True)
        if mode == "full":
            command.add_argument("--workers", type=int, default=1)
        if mode in {"figures", "tables"}:
            command.add_argument("--workspace", type=Path, required=True)
        if mode == "figures":
            command.add_argument("--figure", choices=FIGURE_NAMES, action="append")
        if mode == "exact":
            command.add_argument("--encoding", choices=tuple(EXACT_SPECS), action="append")
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.mode == "test":
        run_script("test_core.py")
        return
    output = args.output.resolve()
    if args.mode in {"figures", "quick", "full"} and "MPLCONFIGDIR" not in os.environ:
        cache = REPO / "outputs" / ".mplconfig"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
    if args.mode == "smoke":
        smoke(output)
    elif args.mode == "exact":
        exact(output, args.encoding or EXACT_SPECS)
    elif args.mode == "figures":
        figures(args.workspace.resolve(), output, args.figure or FIGURE_NAMES)
    elif args.mode == "tables":
        tables(args.workspace.resolve(), output)
    else:
        workers = getattr(args, "workers", 1)
        if workers < 1:
            cli.error("--workers must be positive")
        try:
            reproduction(args.dataset_s2.resolve(strict=True), output, args.mode, workers)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
            cli.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
