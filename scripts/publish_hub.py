"""Phase 8b: stage and publish the model repo and the Space to the Hugging Face Hub.

Staging happens in a temporary folder (nothing is added to this git repo):
    model repo  k08e24bryant/indobert-ewallet-complaints
        /            classifier (config, safetensors, tokenizer) + README.md (model card)
        /bertopic/   demo topic model (BERTopic safetensors export)
    Space       k08e24bryant/ewallet-complaint-analyzer (Gradio, ZeroGPU)
        app.py, demo_core.py, config.yaml (source: hub, cuda, zero_gpu), requirements.txt, README.md,
        figures/*.png (Phase 7 figures used by the Findings tab)

Guards: no review data, labels, parquet/csv/xlsx or pickle files are staged. The HF token is read
by huggingface_hub from the local login and is never printed or written.

Usage:
    uv run python scripts/publish_hub.py --dry-run     # list files and sizes only
    uv run python scripts/publish_hub.py --upload
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "k08e24bryant/indobert-ewallet-complaints"
SPACE_REPO = "k08e24bryant/ewallet-complaint-analyzer"
SPACE_HARDWARE = "zero-a10g"  # ZeroGPU (free personal accounts: up to 2); cpu-basic Gradio Spaces need PRO
STATIC_REPO = "k08e24bryant/ewallet-complaint-findings"  # free static Space (sdk: static)
CLASSIFIER_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]  # no training_args.bin
FORBIDDEN = re.compile(r"\.(parquet|csv|xlsx|xls|bin|pkl|pickle|pt|pth|ckpt|log)$", re.IGNORECASE)


def stage_model(dst: Path) -> None:
    """Classifier at the repo root, topic model in bertopic/, model card as README.md."""
    cfg = yaml.safe_load((ROOT / "app/config.yaml").read_text(encoding="utf-8"))
    clf = (ROOT / "app" / cfg["models"]["classifier"]["local"]).resolve()
    topics = (ROOT / "app" / cfg["models"]["topics"]["local"]).resolve()
    dst.mkdir(parents=True)
    for f in CLASSIFIER_FILES:
        shutil.copy2(clf / f, dst / f)
    shutil.copytree(topics, dst / cfg["models"]["topics"]["hub_subfolder"])
    shutil.copy2(ROOT / "hub/model/README.md", dst / "README.md")


def stage_space(dst: Path) -> None:
    """App code, a hub-sourced config, pinned requirements, README and the Findings figures."""
    dst.mkdir(parents=True)
    for f in ("app.py", "demo_core.py"):
        shutil.copy2(ROOT / "app" / f, dst / f)
    text = (ROOT / "app/config.yaml").read_text(encoding="utf-8")
    replacements = [(r"(?m)^  source: local\b", "  source: hub  "),
                    (r"(?m)^  figures_dir: \.\./reports/figures\b", "  figures_dir: figures"),
                    (r"(?m)^  device: cpu\b", "  device: cuda"),            # ZeroGPU: models on cuda at startup
                    (r"(?m)^  zero_gpu: false\b", "  zero_gpu: true")]
    for pat, rep in replacements:
        text, n = re.subn(pat, rep, text)
        if n != 1:
            raise RuntimeError(f"config.yaml: expected one match for {pat!r}, found {n}")
    (dst / "config.yaml").write_text(text, encoding="utf-8")
    cfg = yaml.safe_load(text)
    assert cfg["models"]["source"] == "hub" and cfg["models"]["hub_repo"] == MODEL_REPO
    assert cfg["models"]["device"] == "cuda" and cfg["models"]["zero_gpu"] is True
    for f in ("README.md", "requirements.txt"):
        shutil.copy2(ROOT / "hub/space" / f, dst / f)
    (dst / "figures").mkdir()
    for name, _ in cfg["findings"]["items"]:
        if not name.endswith(".png"):
            raise RuntimeError(f"figure {name} is not a PNG")
        shutil.copy2(ROOT / "reports/figures" / name, dst / "figures" / name)


def stage_static(dst: Path) -> None:
    """Static findings page: index.html (built by scripts/build_static_space.py), README, Phase 7 PNGs."""
    dst.mkdir(parents=True)
    for f in ("index.html", "README.md"):
        shutil.copy2(ROOT / "hub/static" / f, dst / f)
    (dst / "figures").mkdir()
    page = (dst / "index.html").read_text(encoding="utf-8")
    for name in sorted(set(re.findall(r'src="figures/([^"]+\.png)"', page))):
        shutil.copy2(ROOT / "reports/figures" / name, dst / "figures" / name)


def check_and_list(root: Path, label: str) -> None:
    """Refuse forbidden file types; print every file with its size."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    bad = [p for p in files if FORBIDDEN.search(p.name)]
    if bad:
        raise RuntimeError(f"{label}: forbidden files staged: {[str(p.relative_to(root)) for p in bad]}")
    total = sum(p.stat().st_size for p in files)
    print(f"\n== {label} ({len(files)} files, {total / 2**20:,.1f} MB) ==")
    for p in files:
        size = p.stat().st_size
        print(f"   {str(p.relative_to(root)).replace(chr(92), '/'):<48} {size / 2**20:>10.2f} MB" if size >= 2**20
              else f"   {str(p.relative_to(root)).replace(chr(92), '/'):<48} {size / 1024:>10.1f} KB")


def main() -> None:
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--upload", action="store_true")
    parser.add_argument("--only", choices=["model", "space", "static"], default=None,
                        help="publish one repo only (default: model + live Space; static only when named)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        model_dir, space_dir, static_dir = Path(tmp) / "model", Path(tmp) / "space", Path(tmp) / "static"
        stage_model(model_dir)
        stage_space(space_dir)
        stage_static(static_dir)
        check_and_list(model_dir, f"model repo {MODEL_REPO}")
        check_and_list(space_dir, f"Space {SPACE_REPO}")
        check_and_list(static_dir, f"static Space {STATIC_REPO}")
        if args.dry_run:
            print("\ndry run: nothing uploaded")
            return

        from huggingface_hub import HfApi

        api = HfApi()  # token from the local login; never printed
        if args.only in (None, "model"):
            api.create_repo(MODEL_REPO, repo_type="model", private=False, exist_ok=True)
            info = api.upload_folder(repo_id=MODEL_REPO, repo_type="model", folder_path=str(model_dir),
                                     commit_message="Upload complaint classifier, topic model and model card (Phase 8b)")
            print("model repo commit:", info.commit_url)
        if args.only in (None, "space"):
            api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", space_hardware=SPACE_HARDWARE,
                            private=False, exist_ok=True)
            info = api.upload_folder(repo_id=SPACE_REPO, repo_type="space", folder_path=str(space_dir),
                                     commit_message="Deploy e-wallet complaint analyzer (Phase 8b)")
            print("Space commit:", info.commit_url)
        if args.only == "static":
            api.create_repo(STATIC_REPO, repo_type="space", space_sdk="static", private=False, exist_ok=True)
            info = api.upload_folder(repo_id=STATIC_REPO, repo_type="space", folder_path=str(static_dir),
                                     commit_message="Publish complaint findings page (Phase 8b)")
            print("static Space commit:", info.commit_url)


if __name__ == "__main__":
    main()
