#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
r140-compliant full re-run orchestrator.

Why this exists
---------------
The archived 110 runs in final_log/ were produced *before* the split-policy change
(no [Split] line, run-varying test partition). r140 (ss.4.1, Table 8, App. A.1/A.3)
requires: TEST_SEED=42 fixed test partition shared by EVERY run and EVERY model,
train/val redrawn from the training seed, and paired across models at the same run
index. The training scripts already implement this; this driver re-runs all of them
and rebuilds the archive from the new outputs.

Design (r140 ss.3.4 / 4.1 / 4.2.3 / Table 8)
---------------------------------------------
Phase A (Path A): public ImageNet-1K checkpoint -> target task.
    B1/B2: n=3 (_v0.._v2);  C2/G: n=5 (_v0.._v4)          -> 80 runs
Phase B (Path B): each model's own best Polyp (G) checkpoint -> B1/B2, n=3.
    run i of B1/B2  <-  run i of G, same model               -> 30 runs
    CRITICAL: init comes from the NEW G runs produced by this re-run,
              NOT from the old selected_ckpts/main_G archive.   -> 110 runs total

Every run gets SEED = RUN_SEED[i] and RUN_SUFFIX = i; TEST_SEED=42 is hardcoded
inside each training script and is deliberately never injected by the driver.

Usage
-----
    python rerun_all.py plan                 # print the run plan, no execution
    python rerun_all.py run [--limit N]     # execute Phase A then Phase B (resume-safe)
    python rerun_all.py harvest             # build new archive + verify r140 invariants
    python rerun_all.py verify              # verify an existing re-run only
    python rerun_all.py all                 # run + harvest

Standalone manual launchers run_by_seed.sh / run_pathB.sh implement the same
design; this driver parses their seed table at startup and aborts on any drift.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

EXP = Path("C:/sand_box/exp_script")
SCRIPTS = EXP / "scripts"
TRAIN = SCRIPTS / "train"
CORE = SCRIPTS / "core"
SHELL = SCRIPTS / "shell"
RELEASE = EXP / "indtokenswin_release" / "splits"
ROOT = EXP / "rerun"
MAIN = ROOT / "main"
G_CKPTS = ROOT / "G_ckpts"
CONT = ROOT / "G_continue"
ARCHIVE = ROOT / "archive"
PYTHON = Path("C:/ProgramData/anaconda3/python.exe")

RUN_SEED = (112458, 142956, 212855, 653153, 940839)

# task prefix -> (n runs, final_log group, selected_ckpts group, manifest setting)
TASK_MAP = {
    "ExpB1_": (3, "B1", "main_B1", "B1_hyperkvasir12"),
    "ExpB2_": (3, "B2", "main_B2", "B2_kvasir_capsule12"),
    "ExpC2_": (5, "C2", "main_C2", "C2_siim_isic"),
    "ExpG_": (5, "G", "main_G", "G_polyp_binary"),
}
CONT_MAP = {
    "ExpB1_": (3, "G_continue/B1", "Gc_B1", "B1_hyperkvasir12"),
    "ExpB2_": (3, "G_continue/B2", "Gc_B2", "B2_kvasir_capsule12"),
}

RESULT_FILES = ("test_results.txt", "confusion_matrix.png",
                "confusion_matrix.txt", "predictions.csv")


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def task_name(script: Path) -> str:
    m = re.search(r'^TASK_NAME\s*=\s*["\']([^"\']+)', script.read_text(encoding="utf-8"), re.M)
    if not m:
        raise RuntimeError(f"TASK_NAME not found in {script}")
    return m.group(1)


def check_launcher_sync() -> None:
    """Abort if the standalone shell launchers drifted from the driver's seed table."""
    for sh in ("run_by_seed.sh", "run_pathB.sh"):
        s = (SHELL / sh).read_text(encoding="utf-8")
        m = re.search(r"RUN_SEED=\(([^)]+)\)", s)
        if not m or tuple(int(x) for x in m.group(1).split()) != RUN_SEED:
            raise RuntimeError(f"seed table drift in {sh}: {m.group(1) if m else 'missing'}")
        if "export RESERVE_SEED" in s:
            raise RuntimeError(f"stale RESERVE_SEED export in {sh}")


def build_plan() -> list[dict]:
    scripts = sorted(TRAIN.glob("train_exp*.py"))
    if len(scripts) != 20:
        raise RuntimeError(f"expected 20 training scripts, found {len(scripts)}")
    plan = []
    for sc in scripts:
        tn = task_name(sc)
        fam = next((k for k in TASK_MAP if tn.startswith(k)), None)
        if fam is None:
            raise RuntimeError(f"unknown task family: {tn}")
        n, _, _, _ = TASK_MAP[fam]
        for i in range(n):
            plan.append(dict(phase="A", script=sc, task=tn, idx=i, seed=RUN_SEED[i],
                             out_root=MAIN))
    # Phase B: B1/B2 only, n=3, init from the NEW G runs
    for sc in scripts:
        tn = task_name(sc)
        fam = next((k for k in CONT_MAP if tn.startswith(k)), None)
        if fam is None:
            continue
        n, _, _, _ = CONT_MAP[fam]
        g_task = re.sub(r"^ExpB[12]_", "ExpG_", tn)
        for i in range(n):
            plan.append(dict(phase="B", script=sc, task=tn, idx=i, seed=RUN_SEED[i],
                             out_root=CONT, g_task=g_task))
    return plan


def run_dir(job: dict) -> Path | None:
    """Locate the run dir for a job (date-stamped name, RUN_SUFFIX fixed)."""
    pat = f"{job['task']}_*__v{job['idx']}"
    hits = sorted(job["out_root"].glob(pat))
    return hits[-1] if hits else None


def is_done(job: dict) -> bool:
    d = run_dir(job)
    return bool(d and (d / "test_results.txt").is_file()
                and (d / "checkpoints" / "best_model_phase2.pth").is_file())


def g_init(job: dict) -> Path:
    pat = f"{job['g_task']}_*__v{job['idx']}/best_model_phase2.pth"
    hits = sorted(G_CKPTS.glob(pat))
    if not hits:
        pat2 = f"{job['g_task']}_*__v{job['idx']}/checkpoints/best_model_phase2.pth"
        hits = sorted(G_CKPTS.glob(pat2))
    if len(hits) != 1:
        raise RuntimeError(f"Path B init for {job['task']} _v{job['idx']}: "
                           f"expected exactly 1 G ckpt, got {len(hits)} ({pat})")
    return hits[0]


def launch(job: dict) -> bool:
    env = os.environ.copy()
    env.update(
        KMP_DUPLICATE_LIB_OK="TRUE",
        PYTHONIOENCODING="utf-8",
        PYTHONPATH=os.pathsep.join([str(CORE), str(TRAIN), env.get("PYTHONPATH", "")]),
        SEED=str(job["seed"]),
        RUN_SUFFIX=str(job["idx"]),
        OUTPUT_ROOT=str(job["out_root"]),
    )
    if job["phase"] == "B":
        try:
            env["INIT_FROM"] = str(g_init(job))
        except RuntimeError as e:
            log(f"  SKIP (G init unavailable): {e}")
            return False
    log(f"START {job['phase']} _v{job['idx']} {job['script'].name} "
        f"(SEED={job['seed']}, INIT_FROM={env.get('INIT_FROM', 'ImageNet')})")
    try:
        r = subprocess.run([str(PYTHON), str(job["script"])], env=env, cwd=str(SCRIPTS))
        ok = r.returncode == 0
    except Exception as e:                      # noqa: BLE001
        log(f"  EXCEPTION: {e}")
        return False
    log(f"DONE  {job['phase']} _v{job['idx']} {job['script'].name} "
        f"rc={r.returncode} {'OK' if ok else 'FAILED'}")
    return ok


def do_run(limit: int | None, force: bool) -> None:
    check_launcher_sync()
    for p in (MAIN, CONT, G_CKPTS):
        p.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    log(f"plan: {len(plan)} runs (A={sum(j['phase']=='A' for j in plan)}, "
        f"B={sum(j['phase']=='B' for j in plan)})")

    jobs = [j for j in plan if force or not is_done(j)]
    skipped = len(plan) - len(jobs)
    if skipped:
        log(f"resume: skipping {skipped} already-complete runs")
    if limit:
        jobs = jobs[:limit]

    phaseA = [j for j in jobs if j["phase"] == "A"]
    phaseB = [j for j in jobs if j["phase"] == "B"]
    # G must finish before Path B can resolve its INIT_FROM
    if phaseB and not phaseA:
        if not list(G_CKPTS.glob("*/best_model_phase2.pth")):
            raise RuntimeError("Phase B requested but new G checkpoints are missing; "
                               "run Phase A first, then `harvest --g-only`.")

    failed = []
    for j in phaseA:
        if not launch(j):
            failed.append(j)
    if phaseB:
        build_g_ckpts()
        for j in phaseB:
            if not launch(j):
                failed.append(j)
    log(f"RUN FINISHED: {len(jobs) - len(failed)} ok, {len(failed)} failed")
    for j in failed:
        log(f"  FAILED: {j['phase']} _v{j['idx']} {j['script'].name}")


def build_g_ckpts() -> None:
    """Stage the NEW G best checkpoints under the layout Path B expects."""
    G_CKPTS.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in sorted(MAIN.glob("ExpG_*__v*")):
        src = d / "checkpoints" / "best_model_phase2.pth"
        if not src.is_file():
            continue
        dst = G_CKPTS / d.name / "best_model_phase2.pth"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        n += 1
    log(f"G_ckpts staged: {n} new G best checkpoints")
    if n != 25:
        log(f"WARNING: expected 25 G ckpts (5 models x 5 runs), found {n}")


# ---------------------------------------------------------------- harvest ----
def norm_path(p: str, markers: set[str]) -> str:
    s = p.replace("\\", "/")
    parts = s.split("/")
    for i, seg in enumerate(parts):
        if seg in markers:
            return "/".join(parts[i:])
    return s.rsplit("/", 1)[-1]


def manifest(task: str) -> set[str]:
    setting = next((v[3] for k, v in {**TASK_MAP, **CONT_MAP}.items()
                    if task.startswith(k)), None)
    if setting is None:
        raise RuntimeError(f"no manifest for {task}")
    rows = list(csv.DictReader(io.open(RELEASE / setting / "test_seed_42.csv",
                                      encoding="utf-8")))
    markers = {r["relpath"].split("/")[0] for r in rows}
    return {norm_path(r["relpath"], markers) for r in rows}


def collect(root: Path, mapping: dict, final_log: Path, sel_ckpts: Path) -> list[dict]:
    out = []
    for d in sorted(p for p in root.glob("Exp*__v*") if p.is_dir()):
        if not (d / "test_results.txt").is_file():
            continue
        fam = next((k for k in mapping if d.name.startswith(k)), None)
        if fam is None:
            continue
        _, grp, cgrp, _ = mapping[fam]
        dst = final_log / grp / d.name
        dst.mkdir(parents=True, exist_ok=True)
        for fn in RESULT_FILES:
            if (d / fn).is_file():
                shutil.copy2(d / fn, dst / fn)
        if (d / "logs").is_dir():
            shutil.copytree(d / "logs", dst / "logs", dirs_exist_ok=True)
        ck = d / "checkpoints" / "best_model_phase2.pth"
        if ck.is_file():
            cdir = sel_ckpts / cgrp / d.name
            cdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ck, cdir / "best_model_phase2.pth")
        out.append(dict(run=d.name, group=grp, src=str(d),
                        manifest_group=grp.split("/")[-1]))
    return out


def parse_split_md5(run_dir: Path) -> dict | None:
    logs = sorted((run_dir / "logs").glob("*.log")) if (run_dir / "logs").is_dir() else []
    if not logs:
        logs = list(run_dir.glob("*.log"))
    for lg in logs:
        txt = lg.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"\[Split\].*?MD5\(train/val/test\)=([0-9a-f]+)/([0-9a-f]+)/([0-9a-f]+)", txt)
        if m:
            return dict(train=m.group(1), val=m.group(2), test=m.group(3))
    return None


def verify(runs: list[dict]) -> dict:
    report = dict(total=len(runs), by_group=Counter(), missing_split=[],
                  test_md5_by_group=defaultdict(set), trainval_by_run=defaultdict(set),
                  testset_mismatch=[], counts_ok=True)
    markers = set()
    for setting in ("B1_hyperkvasir12", "B2_kvasir_capsule12",
                    "C2_siim_isic", "G_polyp_binary"):
        rows = csv.DictReader(io.open(RELEASE / setting / "test_seed_42.csv",
                                     encoding="utf-8"))
        markers |= {r["relpath"].split("/")[0] for r in rows}

    pub = {}
    for setting, grp in (("B1_hyperkvasir12", "B1"), ("B2_kvasir_capsule12", "B2"),
                         ("C2_siim_isic", "C2"), ("G_polyp_binary", "G")):
        rows = list(csv.DictReader(io.open(RELEASE / setting / "test_seed_42.csv",
                                          encoding="utf-8")))
        pub[grp] = {norm_path(r["relpath"], markers) for r in rows}

    for r in runs:
        grp = r["group"]
        # G_continue/B1 uses B1's test partition; G_continue/B2 uses B2's
        setting_grp = grp.split("/")[-1] if grp.startswith("G_continue") else grp
        report["by_group"][grp] += 1
        d = Path(r["src"])
        md5 = parse_split_md5(d)
        if not md5:
            report["missing_split"].append(r["run"])
            continue
        report["test_md5_by_group"][setting_grp].add(md5["test"])
        vi = re.search(r"__v(\d+)$", r["run"])
        if vi:
            report["trainval_by_run"][(setting_grp, vi.group(1))].add(
                (md5["train"], md5["val"]))
        pv = d / "predictions.csv"
        if pv.is_file() and setting_grp in pub:
            rows = list(csv.DictReader(io.open(pv, encoding="utf-8", newline="")))
            got = {norm_path(x["image_path"], markers) for x in rows}
            if got != pub[setting_grp]:
                report["testset_mismatch"].append(
                    dict(run=r["run"], got=len(got), want=len(pub[setting_grp]),
                         overlap=len(got & pub[setting_grp])))

    want = {"B1": 15, "B2": 15, "C2": 25, "G": 25,
            "G_continue/B1": 15, "G_continue/B2": 15}
    report["counts_ok"] = dict(report["by_group"]) == want
    report["by_group"] = dict(report["by_group"])
    report["test_md5_by_group"] = {k: sorted(v) for k, v in report["test_md5_by_group"].items()}
    report["trainval_by_run"] = {f"{k[0]}_v{k[1]}": len(v)
                                 for k, v in report["trainval_by_run"].items()}
    return report


def model_of(run: str) -> str:
    s = re.sub(r"_\d{8}(?:_v\d+)?__v\d+$", "", run)
    return re.sub(r"^Exp(?:B1|B2|C2|G)_", "", s)


def build_reports(final_log: Path, reports: Path) -> None:
    """Regenerate the aggregate stats/tables from the NEW runs only."""
    import math
    metrics = defaultdict(lambda: defaultdict(list))
    for tr in sorted(final_log.glob("**/test_results.txt")):
        rel = tr.relative_to(final_log)
        grp = "/".join(rel.parts[:-2])          # B1 | C2 | G | G_continue/B1 ...
        txt = tr.read_text(encoding="utf-8", errors="replace")
        # only scalar metric lines (skip list-valued per_class_* and header text)
        vals = dict(re.findall(r"^\s{2}([a-z][a-z0-9_]*):\s*([0-9.eE+-]+)\s*$", txt, re.M))
        model = model_of(tr.parent.name)
        vi = re.search(r"__v(\d+)$", tr.parent.name)
        key = (grp, model)
        for k, v in vals.items():
            if k.startswith("per_class"):
                continue
            metrics[key][k].append((int(vi.group(1)) if vi else -1, float(v)))

    stats = {}
    for (grp, model), ms in sorted(metrics.items()):
        entry = {}
        for k, pairs in ms.items():
            xs = [v for _, v in sorted(pairs)]
            n = len(xs)
            mean = sum(xs) / n
            sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
            entry[k] = dict(n=n, mean=mean, std_sample=sd, values=xs)
        stats[f"{grp}/{model}"] = entry
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "aggregated_stats_rerun.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    primary = {"B1": "macro_f1", "B2": "macro_f1", "C2": "balanced_accuracy",
               "G": "auc", "G_continue/B1": "macro_f1", "G_continue/B2": "macro_f1"}
    lines = ["# r140 re-run · aggregated results", "",
             f"generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "",
             "> Source: rebuilt final_log (110 runs, TEST_SEED=42 fixed test partition).",
             "> Values are mean +/- sample SD across runs (n=3 for 12-class, n=5 for C2/G).",
             "", "| group | model | selection metric | n | mean +/- SD |",
             "|---|---|---|---|---|"]
    for key, entry in stats.items():
        grp, model = key.rsplit("/", 1)
        m = primary.get(grp)
        if m and m in entry:
            e = entry[m]
            lines.append(f"| {grp} | {model} | {m} | {e['n']} | "
                         f"{e['mean']:.4f} +/- {e['std_sample']:.4f} |")
    (reports / "aggregated_report_rerun.md").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
    log(f"reports rebuilt: {len(stats)} group/model rows")


def do_swap() -> None:
    """Replace the old archive with the new one, gated on r140 invariants."""
    vf = ARCHIVE / "rerun_verification.json"
    if not vf.is_file():
        raise RuntimeError("run `harvest` first")
    rep = json.loads(vf.read_text(encoding="utf-8"))
    problems = []
    if rep["total"] != 110 or not rep["counts_ok"]:
        problems.append(f"run count/coverage: total={rep['total']} counts_ok={rep['counts_ok']}")
    if rep["missing_split"]:
        problems.append(f"{len(rep['missing_split'])} runs lack [Split] MD5 log")
    for grp, md5s in rep["test_md5_by_group"].items():
        if len(md5s) != 1:
            problems.append(f"{grp}: test MD5 not fixed ({len(md5s)} distinct)")
    multi = {k: v for k, v in rep["trainval_by_run"].items() if v > 1}
    if multi:
        problems.append(f"train/val MD5 differs within a run index: {multi}")
    if rep["testset_mismatch"]:
        problems.append(f"{len(rep['testset_mismatch'])} runs off the published test set")
    if problems:
        raise RuntimeError("refusing to swap archive; r140 invariants FAILED:\n  - " +
                           "\n  - ".join(problems))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = EXP / "pre_rerun_archive" / stamp
    bak.mkdir(parents=True, exist_ok=True)
    for name in ("final_log", "selected_ckpts", "reports"):
        src = EXP / name
        if src.exists():
            shutil.move(str(src), str(bak / name))
    shutil.move(str(ARCHIVE / "final_log"), str(EXP / "final_log"))
    shutil.move(str(ARCHIVE / "selected_ckpts"), str(EXP / "selected_ckpts"))
    (EXP / "reports").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ARCHIVE / "rerun_verification.md", EXP / "reports" / "rerun_verification.md")
    shutil.copy2(ARCHIVE / "rerun_verification.json", EXP / "reports" / "rerun_verification.json")
    build_reports(EXP / "final_log", EXP / "reports")
    log(f"archive swapped. old results preserved at {bak}")


def do_harvest(g_only: bool = False) -> None:
    if g_only:
        build_g_ckpts()
        return
    build_g_ckpts()
    final_log = ARCHIVE / "final_log"
    sel_ckpts = ARCHIVE / "selected_ckpts"
    for p in (final_log, sel_ckpts):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)
    runs = collect(MAIN, TASK_MAP, final_log, sel_ckpts)
    runs += collect(CONT, CONT_MAP, final_log, sel_ckpts)
    log(f"harvested {len(runs)} runs into {ARCHIVE}")

    rep = verify(runs)
    (ARCHIVE / "rerun_verification.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# r140 re-run verification", "",
             f"generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "",
             "## Invariants", "",
             f"- total runs: {rep['total']} (expected 110)",
             f"- per-group counts OK: **{rep['counts_ok']}** -> {rep['by_group']}",
             f"- runs missing [Split] MD5 log: {len(rep['missing_split'])}"]
    for grp, md5s in sorted(rep["test_md5_by_group"].items()):
        lines.append(f"- test MD5 within {grp}: {len(md5s)} distinct "
                     f"({'OK - fixed' if len(md5s) == 1 else 'FAIL - varies'})")
    multi = {k: v for k, v in rep["trainval_by_run"].items() if v > 1}
    lines.append(f"- train/val MD5 differing within a run index: {len(multi)} "
                 f"({'OK' if not multi else 'FAIL ' + str(multi)})")
    lines.append(f"- test-set mismatch vs published test_seed_42.csv: "
                 f"{len(rep['testset_mismatch'])}")
    for x in rep["testset_mismatch"][:10]:
        lines.append(f"    - {x['run']}: got {x['got']}, want {x['want']}, "
                     f"overlap {x['overlap']}")
    if rep["missing_split"]:
        lines += ["", "## Missing [Split] logs", ""] + \
                 [f"- {x}" for x in rep["missing_split"][:20]]
    (ARCHIVE / "rerun_verification.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    log("\n".join(lines[3:]))


def do_plan() -> None:
    check_launcher_sync()
    plan = build_plan()
    c = Counter((j["phase"], j["idx"]) for j in plan)
    print(f"total runs: {len(plan)}")
    for (ph, i), n in sorted(c.items()):
        print(f"  phase {ph} _v{i}: {n:2d} runs (SEED={RUN_SEED[i]})")
    print("\nPhase A by group:")
    for k, v in Counter(j["group"] if "group" in j else j["task"].split("_")[0]
                        for j in plan if j["phase"] == "A").items():
        print(f"  {k}: {v}")
    print("\nPath B init mapping (new G runs):")
    for j in plan:
        if j["phase"] == "B":
            print(f"  {j['task']}__v{j['idx']} <- {j['g_task']}__v{j['idx']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "run", "harvest", "verify", "swap", "all"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--g-only", action="store_true")
    a = ap.parse_args()
    if a.cmd == "plan":
        do_plan()
    elif a.cmd == "run":
        do_run(a.limit, a.force)
    elif a.cmd == "harvest":
        do_harvest(a.g_only)
    elif a.cmd == "swap":
        do_swap()
    elif a.cmd == "verify":
        r = verify([dict(run=d.name, group=next(
            (v[1] for k, v in {**TASK_MAP, **CONT_MAP}.items() if d.name.startswith(k)), "?"),
            src=str(d)) for d in list(MAIN.glob("Exp*__v*")) + list(CONT.glob("Exp*__v*"))
            if (d / "test_results.txt").is_file()])
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        do_run(a.limit, a.force)
        do_harvest(False)
        do_swap()
    return 0


if __name__ == "__main__":
    sys.exit(main())
