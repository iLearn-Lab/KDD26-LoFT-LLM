import torch
import numpy as np
from collections import defaultdict
import re
from typing import List, Dict
from layers.GetSolarPrompt import get_prompt
import math
import ast
from tqdm import tqdm

def compute_derived_features_tensor(x: torch.Tensor, cut_len = 30) -> dict:
    """
    输入:
        x: Tensor of shape [B, T, N], where N=13 (columns as described)

    输出:
        dict of {name: tensor of shape [B, T]}
    """
    assert x.shape[-1] >= 13, "Input tensor must have at least 13 features"
    x = x[:, -cut_len:, :]  # 只取最近30个时间步的数据

    tclw = x[:, :, 1]
    tciw = x[:, :, 2]
    r = x[:, :, 4]
    TCC = x[:, :, 5]
    u10 = x[:, :, 6]
    v10 = x[:, :, 7]
    SSRD = x[:, :, 9]
    TP = x[:, :, 12]

    # 1. wind speed
    wind_speed = torch.sqrt(u10**2 + v10**2)

    # 2. effective SSRD (corrected for cloud cover)
    ssrd_eff = SSRD * (1 - TCC)

    # 3. cloud optical thickness proxy
    cloud_opt = 1000 * (tclw + tciw)

    # 4. humidity × cloud
    rh_cloud = r * TCC

    # 5. SSRD delta (along time axis)
    ssrd_delta = torch.zeros_like(SSRD)
    ssrd_delta[:, 1:, ...] = SSRD[:, 1:, ...] - SSRD[:, :-1, ...]

    # 6. suppression score (empirical weights)
    suppression_score = TP * 1000 + r * 0.5 + cloud_opt * 0.05

    return {
        "POWER": x[:, :, 0],
        "wind_speed": wind_speed,
        "ssrd_eff": ssrd_eff,
        "cloud_opt": cloud_opt,
        "rh_cloud": rh_cloud,
        "ssrd_delta": ssrd_delta,
        "suppression_score": suppression_score
    }




def make_prompts_from_fields(field_x, field_y, time_x, time_y, pred_len, 
                             filed_y_high_freq, filed_y_low_freq):
    """
    输入:
        field_x: dict, 历史区间各字段，shape: [B, T]
        field_y: dict, 预测区间各字段，shape: [B, P]
    输出:
        prompts: List[str], 每个batch一个prompt
    """
    # 取batch数
    B = len(time_x)
    
    prompts = []
    for b in range(B):
        prompt = get_prompt(
            history_dates=time_x[b][-30:],
            history_power=field_x['POWER'][b],
            history_ssrd_eff=field_x['ssrd_eff'][b],
            history_cloud_opt=field_x['cloud_opt'][b],
            history_wind_speed=field_x['wind_speed'][b],
            history_rh_cloud=field_x['rh_cloud'][b],
            history_ssrd_delta=field_x['ssrd_delta'][b],
            history_suppression_score=field_x['suppression_score'][b],
            pred_dates=time_y[b],
            predicted_power=field_y['POWER'][b],
            pred_ssrd_eff=field_y['ssrd_eff'][b],
            pred_cloud_opt=field_y['cloud_opt'][b],
            pred_wind_speed=field_y['wind_speed'][b],
            pred_rh_cloud=field_y['rh_cloud'][b],
            pred_ssrd_delta=field_y['ssrd_delta'][b],
            pred_suppression_score=field_y['suppression_score'][b],
            predicted_low_freq=filed_y_low_freq['POWER'][b],
            predicted_high_freq=filed_y_high_freq['POWER'][b],
            pred_len=pred_len
        )

        prompts.append(prompt)
    return prompts


def convert_string_to_float_list(string_list, pred_len):
    # 匹配所有 float 或 int（支持负号）
    numbers = re.findall(r'-?\d+(?:\.\d+)?', string_list)

    try:
        float_numbers = [float(num) for num in numbers]
        # 截断或补齐
        if len(float_numbers) < pred_len:
            float_numbers += [-100.0] * (pred_len - len(float_numbers))
        else:
            float_numbers = float_numbers[:pred_len]
        return float_numbers
    except Exception as e:
        print(f"Error converting to float: {e}")
        return [-100.0] * pred_len



def Qwen(tokenizer, model, prompts, pred_len, power_org, batch_size=8, trues=[]):
    power_refined = []

    num_batches = math.ceil(len(prompts) / batch_size)

    for batch_idx in tqdm(range(num_batches), desc="Qwen Generating"):
        batch_prompts = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # 构建输入文本
        texts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # 禁用“思考模式”
            )
            texts.append(text)

        # 编码并推理
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256
        )

        # 解码与解析
        for i in range(len(generated_ids)):
            output_ids = generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            pred_POWER = convert_string_to_float_list(content, pred_len)
            
            j = batch_idx * batch_size + i
            p_org = power_org[j].cpu().numpy()
            power_np = np.array(pred_POWER)

            # print(power_org.shape)
            for k in range(power_np.shape[0]):
                # if power_np[k] != 0 and np.abs(power_np[k] - p_org[k]) > 1e-4:
                if power_np[k] != 0 and np.abs(power_np[k] - p_org[k]) > 0.2:
                    power_np[k] = p_org[k]
            
            pred_POWER = power_np.tolist()

            # print(batch_prompts[i])
            # print("LLM: ", pred_POWER)
            # print("org: ", p_org)
            # print("trues: ", trues[j])

            power_refined.append(pred_POWER)

    return power_refined


def regroup_predictions_filtered(apply_amt_refined, redeem_amt_refined, fund_ids, feature_names, pred_len):
    """
    输出: tensor [B, P, C2]，C2为feature_names中所有apply/redeem相关列，顺序与feature_names一致
    """
    # 统计所有apply_amt/redeem_amt相关的fund_id及其在fund_ids中的索引
    B = len(fund_ids) // len(set(fund_ids))
    group_per_batch = len(fund_ids) // B

    # 建立fund_id到group idx的映射
    fundid2groupidx = {}
    for b in range(B):
        for g in range(group_per_batch):
            idx = b * group_per_batch + g
            fundid2groupidx[(b, fund_ids[idx])] = idx

    C2 = 0
    col_types = []
    col_fundids = []
    for name in feature_names:
        if name.startswith('apply_amt_'):
            col_types.append('apply')
            col_fundids.append(name.split('_')[-1])
            C2 += 1
        elif name.startswith('redeem_amt_'):
            col_types.append('redeem')
            col_fundids.append(name.split('_')[-1])
            C2 += 1

    out_tensor = torch.zeros(B, pred_len, C2)
    for b in range(B):
        for c, (typ, fundid) in enumerate(zip(col_types, col_fundids)):
            if typ == 'apply':
                key = (b, fundid)
                if key in fundid2groupidx:
                    idx = fundid2groupidx[key]
                    out_tensor[b, :, c] = torch.tensor(apply_amt_refined[idx])
            elif typ == 'redeem':
                key = (b, fundid)
                if key in fundid2groupidx:
                    idx = fundid2groupidx[key]
                    out_tensor[b, :, c] = torch.tensor(redeem_amt_refined[idx])
            # else: 保持为0
    return out_tensor


