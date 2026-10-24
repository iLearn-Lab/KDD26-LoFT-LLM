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
"""
    prompt += (
        "1. **Deep-shade block** 09 : 00 – 20 : 00 local time → POWER = 0.\n"
        "2. **Twilight taper** 08 → 09 must glide down to zero; 20 → 22 must rise "
        "smoothly from zero.\n"
        "3. **Main drivers**\n"
        "   • Positive: one-hour GHI (SSRD_diff / Δt) if > 0  (ρ≈+0.86); 2 m air-temperature "
        "(ρ≈+0.54).\n"
        "   • Negative: RH_cloud (ρ≈–0.55), Cloud_opt & Suppression_score (≈–0.12 ~ –0.23).\n"
        "   • Wind_speed now shows only a weak positive link (ρ≈+0.11).\n"
        "4. **Adjustment logic**\n"
        "   • Raise POWER when GHI rises **and** cloud / suppression signals are low.\n"
        "   • Cut or zero spikes when Cloud_opt or Suppression_score is high, or RH_cloud > 80 %.\n"
        "5. **Continuity** │ΔPOWER│ ≤ 0.33 unless a matching jump in GHI justifies it.\n"
        "6. **Bounds** Clip all outputs to [0, 1] and keep the daylight profile above.\n\n"
    )

    prompt += f"""
    
### Output Format (Strict)

Return the corrected POWER values (length = {pred_len}), all within [0, 1].

⚠️ No explanations, comments, or variable names.

Example:

[{', '.join(['float']*pred_len)}]
"""

    return prompt





