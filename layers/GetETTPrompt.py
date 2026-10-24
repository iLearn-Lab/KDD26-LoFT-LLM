import torch
from typing import List

def _csv_floats(t: torch.Tensor, decimals: int = 1) -> str:
    fmt = f"{{:.{decimals}f}}"
    return "[" + ",".join(fmt.format(x) for x in t.tolist()) + "]"

def _csv_strs(strs: List[str]) -> str:
    return "[" + ",".join(strs) + "]"


# ---------- prompt builder ----------
def get_prompt(
    hist_dates:  List[str],
    hist_HUFL:  torch.Tensor, hist_MUFL: torch.Tensor, hist_LUFL: torch.Tensor,
    hist_HULL:  torch.Tensor, hist_MULL: torch.Tensor,
    hist_OT:    torch.Tensor,
    fut_dates:  List[str],
    fut_HUFL:   torch.Tensor, fut_MUFL: torch.Tensor, fut_LUFL: torch.Tensor,
    fut_HULL:   torch.Tensor, fut_MULL: torch.Tensor,
    raw_OT:     torch.Tensor,
    low_f: torch.Tensor, high_f: torch.Tensor,
    pred_len: int
) -> str:
    """
    Prompt with transformer-centric task overview, softened frequency guidance,
    and core physical rules.  Retains OT, PACTV, REACTV plus reference low/high-f.
    """

    # --- derived drivers ---
    PACTV_hist  = hist_HUFL**2 + hist_MUFL**2 + hist_LUFL**2
    REACTV_hist = hist_HULL    + hist_MULL
    PACTV_fut   = fut_HUFL**2  + fut_MUFL**2  + fut_LUFL**2
    REACTV_fut  = fut_HULL     + fut_MULL

    # ---------- task overview (domain-centric) ----------
    overview = (
        "### TASK OVERVIEW\n"
        "Oil temperature (OT) is the primary indicator of transformer thermal stress and "
        "insulation ageing.  Accurate hourly forecasts enable dynamic over-loading, "
        "prevent hot-spot deterioration, and schedule cooling interventions.  "
        f"The objective is to adjust the provided *PRED_OT* sequence so that the next "
        f"{pred_len} hourly temperatures follow physically sound behaviour given the "
        "electrical loading patterns described below."
    )

    # ---------- physical rules ----------
    rules = (
        "### KEY RULES\n"
        "• Hourly sampling for all series.\n"
        "• PACTV and REACTV from the previous 1–4 h; these delayed loads often dictate the trend.\n"
        "• PACTV = HUFL² + MUFL² + LUFL²  (dominant copper loss).\n"
        "• REACTV = HULL + MULL           (secondary core loss).\n"
        "• PACTV surge → OT peak ≈ +2 h; prolonged dip → trough ≈ +3 h.\n"
        "• ≥ 6 h low PACTV pulls OT toward night baseline.\n"
        "• hourly jumps > 4 °C are unusual, > 8 °C need "
        "load corroboration.\n"
        "• **Minimal-adjustment policy**\n"
        " If evidence is weak, keep the original PRED_OT unchanged."
        "• **Frequency note** — LOW_FREQ and HIGH_FREQ are respectively the low- and "
        "high-frequency components of the model prediction, and they satisfy\n"
        "     `PRED_OT = LOW_FREQ + HIGH_FREQ`.\n"
        "  They are reference values and may be modified as needed while adjusting PRED_OT."
    )

    # ---------- assemble prompt ----------
    prompt = "You are a transformer-oil thermal agent.\n\n" + overview + "\n\n" + rules + "\n\n"

     # HISTORY lists
    prompt += "### HISTORY\n"
    prompt += "DATES="   + _csv_strs(hist_dates)   + "\n"
    prompt += "OT="      + _csv_floats(hist_OT, 3)    + "\n"
    prompt += "PACTV="   + _csv_floats(PACTV_hist) + "\n"
    prompt += "REACTV="  + _csv_floats(REACTV_hist)+ "\n"

    # FUTURE lists
    prompt += "\n### FUTURE_RAW\n"
    prompt += "DATES="   + _csv_strs(fut_dates)    + "\n"
    prompt += "PRED_OT=" + _csv_floats(raw_OT, 3)     + "\n"
    prompt += "PACTV="   + _csv_floats(PACTV_fut)  + "\n"
    prompt += "REACTV="  + _csv_floats(REACTV_fut) + "\n"

    prompt += "\n### FREQ_COMPONENTS (reference)\n"
    prompt += "LOW_FREQ="  + _csv_floats(low_f)   + "\n"
    prompt += "HIGH_FREQ=" + _csv_floats(high_f)  + "\n"

    # OUTPUT
    example = "[" + ", ".join(f"OT_{i}" for i in range(pred_len)) + "]"

    prompt += (
        "\n### OUTPUT_FORMAT\n"
        f"{example}\n"
        "**Return THIS LIST ONLY — no explanations, thinking, reasoning, or analysis.**"
    )
    return prompt
