"""Configuration helpers for the original research entry points (stdlib only)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_profile(path, horizon=None):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("domain") not in {"fundar", "solar"}:
        raise ValueError("domain must be fundar or solar")
    if horizon is not None:
        profile["common"]["pred_len"] = int(horizon)
    common = profile["common"]
    if common["pred_len"] <= 0 or common["seq_len"] <= 0:
        raise ValueError("Sequence and prediction lengths must be positive")
    if not (common["out_patch_len"] > 4 and common["out_patch_len"] <= common["seq_len"]):
        raise ValueError("Original PLFM uses stride=4; require 4 < out_patch_len <= seq_len")
    values = {"domain": profile["domain"], "seq_len": common["seq_len"], "pred_len": common["pred_len"]}

    def expand(value):
        if isinstance(value, str):
            return value.format(**values)
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return expand(profile)


def active_profile(domain):
    path = os.environ.get("LOFT_RELEASE_CONFIG", str(ROOT / "configs" / f"{domain}.json"))
    horizon = os.environ.get("LOFT_HORIZON")
    profile = load_profile(path, horizon)
    if profile["domain"] != domain:
        raise ValueError(f"The active profile does not belong to {domain}")
    return profile


def llm_paths(domain):
    llm = active_profile(domain)["llm"]
    return llm["base_model"], llm["adapter_path"]


def residual_setting(args):
    return '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        "TSLFL_Forcast", args.model_id, args.model, args.features, args.seq_len, args.label_len,
        args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
        args.factor, args.embed, args.distil, args.des, 0)


def sft_runs(args):
    """Use the current configured horizon/model, instead of private checkpoint names."""
    return [(args.pred_len, args.model_id, residual_setting(args))]


def sft_output(domain):
    path = Path(active_profile(domain)["qlora"]["train_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def stage_arguments(profile, stage):
    if stage not in {"plfm", "residual", "sft", "infer"}:
        raise ValueError("Unknown original-code stage")
    arguments = dict(profile["common"])
    arguments.update(profile["stages"][stage])
    if stage == "plfm":
        arguments.pop("TSLFL_patch_len", None)
    # The original data-export parser has no --seed option.
    if stage == "sft":
        arguments.pop("seed", None)
    result = []
    for key, value in arguments.items():
        if isinstance(value, bool):
            raise ValueError(f"Use original parser flags explicitly; boolean profile argument {key} is unsupported")
        result.extend([f"--{key}", str(value)])
    return result
