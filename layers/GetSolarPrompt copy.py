from typing import List, Dict
import torch
import re
from collections import defaultdict
import numpy as np

def get_prompt(
    history_dates: torch.Tensor,
    history_power: torch.Tensor,
    history_ssrd_eff: torch.Tensor,
    history_cloud_opt: torch.Tensor,
    history_wind_speed: torch.Tensor,
    history_rh_cloud: torch.Tensor,
    history_ssrd_delta: torch.Tensor,
    history_suppression_score: torch.Tensor,
    pred_dates: torch.Tensor,
    predicted_power: torch.Tensor,
    pred_ssrd_eff: torch.Tensor,
    pred_cloud_opt: torch.Tensor,
    pred_wind_speed: torch.Tensor,
    pred_rh_cloud: torch.Tensor,
    pred_ssrd_delta: torch.Tensor,
    pred_suppression_score: torch.Tensor,
    predicted_low_freq: torch.Tensor,
    predicted_high_freq: torch.Tensor,
    pred_len: int
) -> str:
    # ---- 0. 前置文字：任务 & 基本度量 ------------------------------
    prompt = (
        f"You are an expert assistant helping improve hourly solar-power forecasts. "
        f"The target variable POWER is the site’s irradiance per unit area, already "
        f"normalised to the range [0, 1] (W m⁻² divided by 1000). We need corrections "
        f"for the next {pred_len} hours.\n\n"
    )

    # ---- 1. 自然语言串联的规则段 ------------------------------
    prompt += (
        "Because this photovoltaic plant sits in a terrain-shadowed valley, sunlight "
        "behaves differently from the usual dawn-to-dusk pattern. From **10 a.m. to "
        "8 p.m. local time the panels lie in deep shade, so POWER should stay "
        "effectively at zero (any value ≤ 0.05 counts as zero). In the twilight hour "
        "leading into this blackout (09 → 10 a.m.) output must glide smoothly down to "
        "zero, and in the first daylight hour on the far side (8 → 9 p.m.) it must "
        "rise just as smoothly from zero. Outside the shaded interval, the hour-to-"
        "hour increment of surface short-wave radiation (**SSRD_eff**) is by far the "
        "strongest driver of generation (ρ ≈ +0.83), closely followed by the net "
        "top-of-atmosphere short-wave increment (**TSR_diff**, ρ ≈ +0.82, embedded in "
        "the suppression score). Thick clouds or high near-surface humidity "
        "(**RH_cloud**, ρ ≈ –0.53) strongly suppress output, while wind speed "
        "(ρ ≈ +0.23) and air temperature (ρ ≈ +0.41) provide weaker positive support, "
        "largely through panel cooling and efficiency shifts. When SSRD_eff climbs "
        "and cloud-related indicators stay low, it is safe to raise the forecast; "
        "when Cloud_opt or the composite **Suppression_score** is high, the forecast "
        "must be scaled down. Keep the power curve realistic: neighbouring hours "
        "should not jump by more than about **0.33** unless a matching surge in "
        "SSRD_eff justifies the spike, and every corrected value must remain within "
        "[0, 1] while following the smooth daylight profile described above.\n\n"
    )

    # ---- 2. 变量释义（一句话控制，避免冗长） ------------------------------
    prompt += (
        "Key drivers in the tables below:\n"
        "SSRD_eff (J m⁻² h⁻¹), Cloud_opt (optical depth index), Wind_speed (m s⁻¹), "
        "RH_cloud (%), SSRD_delta (J m⁻² change), Suppression_score (composite cloud/ "
        "humidity penalty).\n\n"
    )

    # ---- 3. 历史观测 ------------------------------
    prompt += "### Historical observations (1 h resolution)\n"
    for i in range(len(history_dates)):
        prompt += (
            f"- {history_dates[i]} | POWER:{history_power[i]:.4f}, "
            f"SSRD_eff:{history_ssrd_eff[i]:.1f}, Cloud_opt:{history_cloud_opt[i]:.2f}, "
            f"Wind_speed:{history_wind_speed[i]:.2f}, RH_cloud:{history_rh_cloud[i]:.1f}, "
            f"SSRD_delta:{history_ssrd_delta[i]:.1f}, Suppression:{history_suppression_score[i]:.2f}\n"
        )
    prompt += "\n"

    # ---- 4. 未来初始预测 ------------------------------
    prompt += f"### Preliminary forecast (next {pred_len} h)\n"
    for i in range(len(pred_dates)):
        prompt += (
            f"- {pred_dates[i]} | Pred_POWER:{predicted_power[i]:.4f}, "
            f"SSRD_eff:{pred_ssrd_eff[i]:.1f}, Cloud_opt:{pred_cloud_opt[i]:.2f}, "
            f"Wind_speed:{pred_wind_speed[i]:.2f}, RH_cloud:{pred_rh_cloud[i]:.1f}, "
            f"SSRD_delta:{pred_ssrd_delta[i]:.1f}, Suppression:{pred_suppression_score[i]:.2f}\n"
        )
    prompt += "\n"

    prompt += f"""\n---

### Frequency Component Reference

For each predicted value, the model also provides two internal signal components:

- `predicted_power = low_freq + high_freq`
These components are **not actual cash flows**, but help explain the forecast's structure.

- `low_freq`: trend component — [{', '.join(f'{v.item():.3f}' for v in predicted_low_freq)}]
- `high_freq`: fluctuation component — [{', '.join(f'{v.item():.3f}' for v in predicted_high_freq)}]

---

### Your Task

You must **adjust** the predicted POWER values considering:

- **No power generation between 10:00–20:00** → set to **0**.
- Power **near sunset (10:00) and sunrise (20:00)** should be **significantly lower** than mid-day.
- Leveraging causal relationships between inputs (e.g., SSRD_eff ↑ ⇒ POWER ↑).
- Suppressing over-predicted `Pred_PWR` values when Cloud_Opt or Suppression_Score is high.
- Recognizing low-frequency patterns (baseline shape) and justifying high-frequency spikes only when meteorological variables support them.
- Enforcing physical plausibility: output must be **continuous**, **bounded between 0 and 1**, and **consistent with known irradiance behavior**.

### Output Format (Strict)

Return the corrected POWER values (length = {pred_len}), all within [0, 1].

⚠️ No explanations, comments, or variable names.

Example:

[{', '.join(['float']*pred_len)}]
"""

    return prompt





