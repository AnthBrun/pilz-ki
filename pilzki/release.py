"""Step 8: publish an export as a GitHub release and pin it in pilzdaten.

    python -m pilzki release --export work/export/ft-fungi-effv2b3-final
    python -m pilzki release --export ... --dry-run          # show what would happen

1. checks the export (meta.json with version and gate, model.int8.onnx),
2. creates the GitHub release `model-<version>` in this repo with meta.json,
   model.int8.onnx and SHA256SUMS; the notes carry the run's report and gate
   report, so every model version documents its own numbers,
3. writes `model.lock` (version, download URL, sha256 of both files) into a
   pilzdaten checkout on a branch `model/<version>` and opens a pull request.

pilzdaten's Docker build downloads exactly the locked files and checks their
sha256 (scripts/model.sh there), so deploying a new model = merging that PR
and rebuilding. Rolling back = reverting the lock commit.
Needs the GitHub CLI (`gh`) logged in with write access to both repos.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

from .common import REPO, work_dir

LOCK_NAME = "model.lock"
FILES = ("meta.json", "model.int8.onnx")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cmd: list[str], cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def github_repo(path: Path) -> str:
    """owner/name from the origin remote of a checkout."""
    url = run(["git", "remote", "get-url", "origin"], cwd=path, capture=True)
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    if not match:
        raise SystemExit(f"{path}: origin {url!r} is not a GitHub repository")
    return match.group(1)


def check_export(export: Path, allow_no_gate: bool) -> dict:
    for name in FILES:
        if not (export / name).exists():
            raise SystemExit(f"{export / name} missing; run `pilzki export` first")
    meta = json.loads((export / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("version"):
        raise SystemExit("meta.json has no version")
    if not meta.get("gate") and not allow_no_gate:
        raise SystemExit("export has no gate (run `pilzki train-gate` and export again) or pass --allow-no-gate")
    return meta


def release_notes(meta: dict) -> str:
    run_dir = work_dir() / "runs" / meta["run"]
    lines = [f"Bildmodell für die Foto-Vorschläge in pilzdaten, Run `{meta['run']}`, Backbone `{meta['backbone']}`.", ""]
    if meta.get("gate"):
        lines.append(f"Gate gegen Fotos ohne Pilz: Schwelle {meta['gate']['threshold']:.4f}.")
    lines += ["", "Einbinden: den PR in pilzdaten mergen (setzt `model.lock`), dann `docker compose up -d --build`.", ""]
    report = run_dir / "report.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        lines += [text.split("## Tags")[0].strip(), ""]  # per-tag table is long; it stays in the run dir
    gate = run_dir / "gate_report.md"
    if gate.exists():
        lines += [gate.read_text(encoding="utf-8").strip(), ""]
    return "\n".join(lines)


def lock_text(meta: dict, sums: dict[str, str], download: str, previous: str) -> str:
    """model.lock for pilzdaten: shell-sourceable KEY=value; onnxruntime-web lines are kept."""
    kept = [line for line in previous.splitlines() if line.startswith("ORT_")]
    return "\n".join([
        "# Bildmodell für die Foto-Vorschläge. Geschrieben von `pilzki release` im Repo pilz-ki;",
        "# scripts/model.sh lädt die Dateien und prüft die Prüfsummen (auch im Docker-Build).",
        f"MODEL_VERSION={meta['version']}",
        f"MODEL_URL={download}",
        f"META_SHA256={sums['meta.json']}",
        f"ONNX_SHA256={sums['model.int8.onnx']}",
        *kept,
        "",
    ])


def main(args) -> None:
    export = Path(args.export)
    meta = check_export(export, args.allow_no_gate)
    version = meta["version"]
    tag = f"model-{version}"
    repo = args.repo or github_repo(REPO)
    sums = {name: sha256(export / name) for name in FILES}
    download = f"https://github.com/{repo}/releases/download/{tag}"
    notes = release_notes(meta)
    print(f"release {tag} in {repo}: " + ", ".join(f"{n} {s[:12]}" for n, s in sums.items()))
    if args.dry_run:
        print(notes)
    else:
        exists = subprocess.run(["gh", "release", "view", tag, "--repo", repo], capture_output=True).returncode == 0
        if exists:
            raise SystemExit(f"release {tag} exists already; export a new version instead of overwriting")
        head = run(["git", "rev-parse", "HEAD"], cwd=REPO, capture=True)
        pushed = run(["git", "branch", "-r", "--contains", head], cwd=REPO, capture=True)
        if not pushed:
            raise SystemExit("pilz-ki HEAD is not pushed; push first so the release tag points to published code")
        with tempfile.TemporaryDirectory() as tmp:
            sums_file = Path(tmp) / "SHA256SUMS"
            sums_file.write_text("".join(f"{s}  {n}\n" for n, s in sums.items()))
            notes_file = Path(tmp) / "notes.md"
            notes_file.write_text(notes, encoding="utf-8")
            run(["gh", "release", "create", tag, "--repo", repo, "--target", head, "--title", f"Bildmodell {version}",
                 "--notes-file", str(notes_file), *(str(export / n) for n in FILES), str(sums_file)])
        print(f"released https://github.com/{repo}/releases/tag/{tag}")

    if not args.pilzdaten:
        return
    pilzdaten = Path(args.pilzdaten)
    lock = pilzdaten / LOCK_NAME
    previous = lock.read_text(encoding="utf-8") if lock.exists() else ""
    text = lock_text(meta, sums, download, previous)
    if "ORT_VERSION=" not in text:
        raise SystemExit(f"{lock} has no ORT_VERSION/ORT_SHA256 lines to keep; add them in pilzdaten first")
    if args.dry_run:
        print(f"--- {lock}\n{text}")
        return
    if args.no_pr:
        lock.write_text(text, encoding="utf-8")
        print(f"wrote {lock} (no branch, no PR)")
        return
    if run(["git", "status", "--porcelain"], cwd=pilzdaten, capture=True):
        raise SystemExit(f"{pilzdaten} has uncommitted changes; commit or stash them first")
    branch = f"model/{version}"
    run(["git", "fetch", "origin", args.pilzdaten_base], cwd=pilzdaten)
    run(["git", "checkout", "-b", branch, f"origin/{args.pilzdaten_base}"], cwd=pilzdaten)
    lock.write_text(text, encoding="utf-8")
    run(["git", "add", LOCK_NAME], cwd=pilzdaten)
    run(["git", "-c", "commit.gpgsign=false", "commit", "-m", f"chore(model): Bildmodell {version}\n\nRelease: https://github.com/{repo}/releases/tag/{tag}"], cwd=pilzdaten)
    run(["git", "push", "-u", "origin", branch], cwd=pilzdaten)
    body = f"Setzt das Bildmodell auf [{tag}](https://github.com/{repo}/releases/tag/{tag}).\n\n" + notes
    run(["gh", "pr", "create", "--repo", github_repo(pilzdaten), "--base", args.pilzdaten_base, "--head", branch,
         "--title", f"Bildmodell {version}", "--body", body], cwd=pilzdaten)


def add_arguments(p) -> None:
    p.add_argument("--export", required=True, help="export dir (meta.json, model.int8.onnx)")
    p.add_argument("--repo", help="GitHub repo for the release (default: origin of this checkout)")
    p.add_argument("--pilzdaten", default=str(REPO.parent / "pilzdaten"), help="pilzdaten checkout to update ('' = skip)")
    p.add_argument("--pilzdaten-base", default="main", help="branch the model PR targets")
    p.add_argument("--no-pr", action="store_true", help="only write model.lock into the pilzdaten checkout")
    p.add_argument("--allow-no-gate", action="store_true")
    p.add_argument("--dry-run", action="store_true")
