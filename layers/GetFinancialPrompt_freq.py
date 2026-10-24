from typing import List, Dict
import torch
import re
from collections import defaultdict
import numpy as np

def compute_uv_texts(
    uv_fundown: torch.Tensor,
    uv_stableown: torch.Tensor,
    uv_fundopt: torch.Tensor,
    uv_fundmarket: torch.Tensor,
    eps: float = 1e-6
) -> list[str]:
    uv_texts = []
    for i in range(uv_fundown.shape[0]):
        total_uv = (
            uv_fundown[i] + uv_stableown[i] +
            uv_fundopt[i] + uv_fundmarket[i] + eps
        )
        buy_ratio = uv_fundopt[i] / total_uv
        uv_texts.append(
            f"Total UV: {int(total_uv.item())}, "
            f"Selected-fund preference ratio: {buy_ratio.item():.2f}"
        )
    return uv_texts


def get_low_prompt(
    fund_id: str,
    dates: torch.Tensor,
    apply_amt: torch.Tensor,             # shape [T]
    redeem_amt: torch.Tensor,            # shape [T]
    yield_: torch.Tensor,                # shape [T]
    uv_fundown: torch.Tensor,
    uv_stableown: torch.Tensor,
    uv_fundopt: torch.Tensor,
    uv_fundmarket: torch.Tensor,
    is_month_end: torch.Tensor,          # int tensor [T]
    is_trade: torch.Tensor,              # int tensor [T]
    pred_dates: torch.Tensor,
    predicted_apply: torch.Tensor,       # shape [pred_len]
    predicted_redeem: torch.Tensor,      # shape [pred_len]
    pred_yield_: torch.Tensor,           # shape [pred_len]
    pred_uv_fundown: torch.Tensor,
    pred_uv_stableown: torch.Tensor,
    pred_uv_fundopt: torch.Tensor,
    pred_uv_fundmarket: torch.Tensor,
    pred_is_month_end: torch.Tensor,     # int tensor [pred_len]
    pred_is_trade: torch.Tensor,         # int tensor [pred_len]
    pred_len: int
) -> str:
    uv_texts = compute_uv_texts(uv_fundown, uv_stableown, uv_fundopt, uv_fundmarket)
    pred_uv_texts = compute_uv_texts(pred_uv_fundown, pred_uv_stableown, pred_uv_fundopt, pred_uv_fundmarket)

    prompt = f"""### Role: Financial Sequence Analyst

You are a quantitative analyst focusing on mutual fund capital flow modeling.
Your task is to **refine the predicted daily subscription (`apply_amt`) and redemption (`redeem_amt`) values for the next {pred_len} calendar days** for mutual fund **{fund_id}**, leveraging historical transaction behavior, macroeconomic signals, and behavioral intent derived from user interaction data.

---

### Domain Context & Data Preprocessing

All `apply_amt` and `redeem_amt` time series values below have been pre-processed using **FFT-based low-pass filtering**, preserving **long-term structural trends** while suppressing high-frequency noise. This method is commonly used in professional quantitative finance to reduce short-term volatility and highlight macro-patterns such as investment flows, liquidity cycles, and periodic behaviors (e.g., month-end rebalancing).

⚠️ Please note:
- The values you see for `apply_amt` and `redeem_amt` reflect **only the low-frequency components** of historical and predicted cash flow behavior.
- All other features (e.g., yields, UV metrics, trading flags) are preserved in **original, high-resolution** form to support semantic interpretation.

---

### Feature Description

- `apply_amt` (low-pass): Net subscriptions into the fund, denoised using FFT low-pass filtering.
- `redeem_amt` (low-pass): Net redemptions from the fund, FFT filtered.
- `yield`: 1-year interbank CD yield (%), a proxy for prevailing short-term interest rates and opportunity cost of capital. Higher yields may encourage redemptions.
- `is_trade`: 1 = valid trading day, 0 = non-trading day (no flow should occur).
- `is_month_end`: 1 = last trading day of the month, often associated with portfolio rebalancing, performance harvesting, and behavioral anomalies.
- `Total UV`: Aggregated user pageviews related to fund-related content. This represents the overall level of investor attention or engagement with the fund domain.
- `Selected-fund preference ratio` = `uv_fundopt / total_uv`: A normalized indicator of buy-side intent. A higher ratio suggests concentrated interest in purchasing specific funds, often predictive of net inflows.
---

### Historical Trading Days (Low-Frequency Net Flow Observed)

Recent historical activity for fund `{fund_id}` (filtered for structural trends only):
"""
    for i in range(len(dates)):
        if is_trade[i].item() == 0:
            continue
        prompt += (
            f"- {dates[i].item()} | apply: {apply_amt[i].item():.3f}, redeem: {redeem_amt[i].item():.3f}, "
            f"yield: {yield_[i].item():.6f}, {uv_texts[i]}, month_end: {is_month_end[i].item()}\n"
        )

    prompt += f"""

---

### Prediction Window: Next {pred_len} Calendar Days

The following `apply_amt` and `redeem_amt` values are **model-generated predictions**, pre-processed by low-pass filtering to retain long-term capital flow structure.
These are **not observed values**, but forecasted structural flows.
Your task is to refine and correct them based on professional insight and contextual financial cues.

Your job is to apply financial judgment to refine these smoothed forecasts.
"""
    for i in range(pred_len):
        prompt += (
            f"- {pred_dates[i].item()} | predicted apply: {predicted_apply[i].item():.3f}, "
            f"predicted redeem: {predicted_redeem[i].item():.3f}, "
            f"yield: {pred_yield_[i].item():.6f}, {pred_uv_texts[i]}, "
            f"month_end: {pred_is_month_end[i].item()}, is_trade: {pred_is_trade[i].item()}\n"
        )

    prompt += f"""

---

### Instructions:

You are given **low-pass filtered model predictions** for the next {pred_len} calendar days.
These are **forecasted values**, not ground-truth data.
Your role is to correct and improve them using financial logic and domain-specific reasoning.

Please:
- Refine `apply_amt` and `redeem_amt` to better reflect:
    - directional intent shifts (UV behavior),
    - macroeconomic incentives (interest rates),
    - seasonal/structural drivers (e.g., month-end effect),
    - and previous trend-consistent capital flows.

✅ Your output must remain **low-frequency in nature**. Do not reintroduce daily volatility or noise.
⚠️ Do not add abrupt day-level fluctuations. Preserve smoothness, coherence, and macro patterns.
Use domain cues (e.g., rising yields, rebalancing windows) to **gently** adjust the predicted trend.
Think in terms of capital allocation behavior over time, not transactional noise.

Professional guidelines:
- If `is_trade == 0`, both apply and redeem must be **exactly 0.0**.
- Higher **selected-fund preference ratio** → typically implies inflow strength.
- Elevated **yield** → may signal outflows toward safer assets.
- Month-end often reflects **allocation inflows** (e.g., salary investment plans) or **profit-taking exits**.

💼 Treat this as a structural signal adjustment task — focusing on **smooth, interpretable capital dynamics**.

---

### Output Format (Strict)

Please return **exactly two lists of {pred_len} float values**, representing:

1. Refined `apply_amt` values
2. Refined `redeem_amt` values

⚠️ Output only the two lists. No comments, no explanation, no keys.

Example:

[{', '.join(['float']*pred_len)}]
[{', '.join(['float']*pred_len)}]
"""
    return prompt




def get_high_prompt(
    fund_id: str,
    dates: torch.Tensor,
    apply_amt: torch.Tensor,
    redeem_amt: torch.Tensor,
    yield_: torch.Tensor,
    uv_fundown: torch.Tensor,
    uv_stableown: torch.Tensor,
    uv_fundopt: torch.Tensor,
    uv_fundmarket: torch.Tensor,
    is_month_end: torch.Tensor,
    is_trade: torch.Tensor,
    pred_dates: torch.Tensor,
    predicted_apply: torch.Tensor,
    predicted_redeem: torch.Tensor,
    pred_yield_: torch.Tensor,
    pred_uv_fundown: torch.Tensor,
    pred_uv_stableown: torch.Tensor,
    pred_uv_fundopt: torch.Tensor,
    pred_uv_fundmarket: torch.Tensor,
    pred_is_month_end: torch.Tensor,
    pred_is_trade: torch.Tensor,
    pred_len: int
) -> str:
    uv_texts = compute_uv_texts(uv_fundown, uv_stableown, uv_fundopt, uv_fundmarket)
    pred_uv_texts = compute_uv_texts(pred_uv_fundown, pred_uv_stableown, pred_uv_fundopt, pred_uv_fundmarket)

    prompt = f"""### Role: Financial Sequence Analyst

You are a quantitative analyst focusing on **short-term capital flow fluctuations** in mutual funds.
Your task is to **refine the predicted daily subscription (`apply_amt`) and redemption (`redeem_amt`) values for the next {pred_len} calendar days** for mutual fund **{fund_id}**, focusing on short-term behavioral corrections and market microstructure effects.

---

### Domain Context & Signal Processing

All `apply_amt` and `redeem_amt` time series values below have been pre-processed using **FFT-based high-pass filtering**, extracting **short-term fluctuations** and **residual noise-like components** from the fund's capital flow behavior.

📈 In financial modeling, high-frequency signals are often used to analyze:
- liquidity shocks,
- T+0/T+1 fund rotation,
- intra-week behavioral volatility,
- transient reactions to macro/micro news,
- market microstructure anomalies (e.g., redemption spikes, panic flows).

⚠️ Please note:
- The provided `apply_amt` and `redeem_amt` contain **only the high-frequency components** of historical and predicted cash flow behavior.
- These residual signals are layered over long-term structure, and should be adjusted without distorting their short-term character.
- Contextual features (`yield`, UV, trade flags) remain in full resolution to guide micro-adjustment logic.

---

### Feature Description

- `apply_amt` (high-pass): Day-level net subscription fluctuation component, isolated via FFT high-pass filtering.
- `redeem_amt` (high-pass): Day-level redemption fluctuations, also high-pass filtered.
- `yield`: 1-year interbank CD yield (%), indicating capital cost and macro rate environment. Sudden rate shifts may trigger redemptions or reallocations.
- `is_trade`: 1 = trading day, 0 = market closed.
- `is_month_end`: 1 = month-end rebalancing day, possible source of liquidity bursts or timing anomalies.
- `Total UV`: Aggregated fund-related pageviews, proxying market attention level.
- `Selected-fund preference ratio` = `uv_fundopt / total_uv`: Reflects net buying sentiment or alpha-seeking behavior in short windows.

---

### Historical Trading Days (High-Frequency Net Flow Residuals)

Recent observed short-term fluctuations for fund `{fund_id}`:
"""
    for i in range(len(dates)):
        if is_trade[i].item() == 0:
            continue
        prompt += (
            f"- {dates[i].item()} | apply: {apply_amt[i].item():.3f}, redeem: {redeem_amt[i].item():.3f}, "
            f"yield: {yield_[i].item():.6f}, {uv_texts[i]}, month_end: {is_month_end[i].item()}\n"
        )

    prompt += f"""

---

### Prediction Window: Next {pred_len} Calendar Days

The following predicted values are outputs of a base model and reflect **forecasted short-term fluctuations** (not observed values).
These are **not ground truth** — they require expert post-adjustment based on financial signals and short-term behavior.

Please refine them based on behavior-driven, macro-sensitive, and structural micro-patterns:
"""
    for i in range(pred_len):
        prompt += (
            f"- {pred_dates[i].item()} | predicted apply: {predicted_apply[i].item():.3f}, "
            f"predicted redeem: {predicted_redeem[i].item():.3f}, "
            f"yield: {pred_yield_[i].item():.6f}, {pred_uv_texts[i]}, "
            f"month_end: {pred_is_month_end[i].item()}, is_trade: {pred_is_trade[i].item()}\n"
        )

    prompt += f"""

---

### Instructions:

You are given **high-pass filtered predictions** (`apply_amt`, `redeem_amt`) for the next {pred_len} calendar days.
These are **model-generated estimates**, not real market data.
Your job is to adjust and improve them based on professional financial reasoning.

Your job:
- Refine the predicted short-term flows using:
    - high-sensitivity to UV-based behavioral shifts,
    - market noise reactions (especially to rates),
    - structural timing anomalies (e.g., month-end spikes),
    - and localized reallocation patterns.

✅ Output must retain high-frequency structure — do **not** smooth or average them.
⚠️ These predictions may reflect noise, overreaction, or lag — your role is to correct them using real-world insight.
This is a **fine-tuning task** on short-term forecast residuals, not a direct data recovery task.
Preserve **signal sharpness**, but eliminate irrational spikes or behavioral inconsistencies.

Financial guidance:
- If `is_trade == 0`, both apply and redeem should be **0.0**.
- Sharp increases in `selected-fund preference ratio` may cause **short-lived apply surges**.
- Abrupt yield spikes → **panic redemptions** or **short-term switching**.
- Month-end may cause **capital parking**, **temporary liquidity drainage**, or **surge flows before cutoff windows**.

---

### Output Format (Strict)

Please return **exactly two lists of {pred_len} float values**, representing:

1. Refined high-frequency `apply_amt` values
2. Refined high-frequency `redeem_amt` values

⚠️ Output only the two lists. No comments, no explanation, no keys.

Example:

[{', '.join(['float']*pred_len)}]
[{', '.join(['float']*pred_len)}]
"""
    return prompt
