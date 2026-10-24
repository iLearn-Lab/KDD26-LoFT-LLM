import torch
import numpy as np
from collections import defaultdict
import re
from typing import List, Dict
from layers.GetETTPrompt import get_prompt
import math
import ast
from tqdm import tqdm

def compute_derived_features_tensor(x: torch.Tensor) -> dict:
    """
    输入:
        x: Tensor of shape [B, T, N], where N=13 (columns as described)

    输出:
        dict of {name: tensor of shape [B, T]}
    """
    # assert x.shape[-1] >= 13, "Input tensor must have at least 13 features"
    x = x[:, -30:, :]  # 只取最近30个时间步的数据

    OT = x[:, :, 0]
    HUFL = x[:, :, 1]
    HULL = x[:, :, 2]
    MUFL = x[:, :, 3]
    MULL = x[:, :, 4]
    LUFL = x[:, :, 5]
    LULL = x[:, :, 6]

    return {
        "OT": OT,
        "HUFL": HUFL,
        "HULL": HULL,
        "MUFL": MUFL,
        "MULL": MULL,
        "LUFL": LUFL,
        "LULL": LULL
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
            hist_dates=time_x[b][-30:],
            hist_OT=field_x['OT'][b],
            hist_HUFL=field_x['HUFL'][b],
            hist_HULL=field_x['HULL'][b],
            hist_MUFL=field_x['MUFL'][b],
            hist_MULL=field_x['MULL'][b],
            hist_LUFL=field_x['LUFL'][b],
            fut_dates=time_y[b][-30:],
            fut_HUFL=field_y['HUFL'][b],
            fut_HULL=field_y['HULL'][b],
            fut_MUFL=field_y['MUFL'][b],
            fut_MULL=field_y['MULL'][b],
            fut_LUFL=field_y['LUFL'][b],
            raw_OT=field_y['OT'][b],
            low_f=filed_y_low_freq['OT'][b],
            high_f=filed_y_high_freq['OT'][b],
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
            float_numbers += [10.0] * (pred_len - len(float_numbers))
        else:
            float_numbers = float_numbers[:pred_len]
        return float_numbers
    except Exception as e:
        print(f"Error converting to float: {e}")
        return [10.0] * pred_len



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
            if np.mean(np.abs(power_np - p_org)) > 10: pred_POWER = p_org.tolist()

            # print(batch_prompts[i])
            # print("LLM: ", content)
            # print("org: ", p_org)
            # print("trues: ", trues[i])

            power_refined.append(pred_POWER)

    return power_refined





