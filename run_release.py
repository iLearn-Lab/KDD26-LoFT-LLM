"""Launch the existing entry points with JSON configuration; no model rewrites."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from release_config import ROOT, load_profile, stage_arguments

ENTRIES = {"plfm": "run_plfm.py", "residual": "run_TSLFL.py",
           "sft": "run_TSLFL_LLM_data.py", "infer": "run_TSLFL.py"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=[*ENTRIES, "qlora"])
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print the original entry and arguments without importing ML libraries")
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    profile = load_profile(path, args.horizon)
    if args.stage == "qlora":
        command = [sys.executable, "train_qlora.py", "--config", str(path),
                   "--horizon", str(profile["common"]["pred_len"])]
    else:
        command = [sys.executable, ENTRIES[args.stage], *stage_arguments(profile, args.stage)]
    print(json.dumps({"domain": profile["domain"], "command": command}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env["LOFT_RELEASE_CONFIG"] = str(path)
    env["LOFT_HORIZON"] = str(profile["common"]["pred_len"])
    if profile.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(profile["cuda_visible_devices"])
    return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
