# LoFT-LLM

Code for **LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models**.
[Paper](https://arxiv.org/abs/2512.20002v3) · [中文说明](README_zh.md) · [Method](#method-overview) · [Getting started](#installation) · [Citation](#citation)

LoFT-LLM is a time-series forecasting framework for financial and energy applications with limited training data. It combines low-frequency trend prediction, high-frequency residual learning, and LLM calibration guided by auxiliary context and domain knowledge.

## Motivation

With limited historical data, high-frequency noise can obscure stable temporal patterns. Auxiliary variables, such as user activity or weather conditions, also carry useful context that numerical representations may underuse. LoFT-LLM addresses both challenges through frequency-aware learning and structured natural-language prompts.

## Method overview

The framework uses three training stages:

| Stage | Module | Approach |
| --- | --- | --- |
| 1. Low-frequency learning | Patch Low-Frequency Forecasting Module (PLFM) | Apply FFT to overlapping historical patches and model real and imaginary components with separate MLPs. Use low-pass targets and Frequency Alignment Loss (FALoss) for supervision. |
| 2. Residual learning | iTransformer | Freeze PLFM and learn residuals from high-pass-filtered history. Add trend and residual forecasts and train with FALoss. |
| 3. Semantic calibration | Qwen3-8B with QLoRA | Build domain prompts from history, numerical forecasts and auxiliary context. Fine-tune the LLM to generate calibrated numerical predictions. |

The training and inference pipeline is shown below:

![LoFT-LLM training and inference pipeline](https://arxiv.org/html/2512.20002v3/Training_Process.png)

*Figure 1 from [LoFT-LLM](https://arxiv.org/html/2512.20002v3), by Jiacheng You et al. ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).*

## Tasks and evaluation

| Dataset | Domain | Sampling | Forecast target |
| --- | --- | --- | --- |
| FundAR | Finance | Daily | Fund subscription and redemption flows |
| Solar | Energy | Hourly | Solar power generation |

The [paper](https://arxiv.org/html/2512.20002v3) evaluates full-data and few-shot forecasting, component ablations, and low-frequency learning. The configurations below cover both domains.

## Installation

Use Python 3.10 and a suitable CUDA/PyTorch installation, then install dependencies:

```bash
pip install -r requirements.txt
```

## Data and configuration

Edit `configs/fundar.json` or `configs/solar.json` to set dataset paths, visible CUDA devices, forecast length and model paths. The base model is [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), or a local copy. Datasets and trained adapters are obtained or generated separately.

Set `common.root_path` to the dataset directory and `stages.<stage>.data_path` to the corresponding CSV filename. The loaders expect the following formats:

| Dataset | File | Columns |
| --- | --- | --- |
| FundAR | `ProductAmt.csv` | `date`, followed by 88 numerical target columns |
| FundAR | `ProductAmt_with_info.csv` | `date`, the same 88 target columns in the same order, then auxiliary information |
| Solar | `Solar.csv` | `date`, followed by one power target column |
| Solar | `Solar_Info.csv` | `date`, the power target, then meteorological features; at least 13 numerical columns in total |

FundAR uses 44 funds with `apply_amt_<fund ID>` and `redeem_amt_<fund ID>` targets. Auxiliary fields include `yield_`, `uv_fundown`, `uv_stableown`, `uv_fundopt`, `uv_fundmarket`, `is_trade` and `is_month_end`; names and ordering must match `layers/FinancialLLMProcess_utils.py`. The dataset comes from the [Tianchi fund flow forecasting competition](https://tianchi.aliyun.com/competition/entrance/532224/information).

Solar features are read by position in `layers/SolarLLMProcess_utils.py`. With the target at numerical column 0, required positions include 1=`tclw`, 2=`tciw`, 4=`r`, 5=`TCC`, 6=`u10`, 7=`v10`, 9=`SSRD` and 12=`TP`. Preserve the complete feature order expected by that function and match the prompt's time zone, units and normalization.

Supervised examples use the training partition. For inference, supply auxiliary information that is available at forecast time.

## Training and inference

Run the stages in order from this directory:

```bash
# Stage 1: PLFM low-frequency learning
python run_release.py --config configs/fundar.json --stage plfm
# Stage 2: residual learning with a frozen PLFM
python run_release.py --config configs/fundar.json --stage residual
# Stage 3: export supervised examples and fine-tune Qwen3-8B
python run_release.py --config configs/fundar.json --stage sft
python run_release.py --config configs/fundar.json --stage qlora
# LLM calibration and evaluation
python run_release.py --config configs/fundar.json --stage infer
```

Replace `fundar` with `solar` for Solar. Use the same forecast length across all stages; for example, add `--horizon 3` to each command. Use `--dry-run` to print the command and arguments without loading models.

Supervised examples are written to `outputs/<domain>/h<horizon>/` and adapters to `adapters/<domain>/h<horizon>/`. Checkpoint paths and QLoRA hyperparameters are configured in the domain JSON file.

## Checks

```bash
python -m unittest discover -s tests -v
# Validate supervised examples after export:
python train_qlora.py --config configs/fundar.json --check-data
```

## Citation

If you use LoFT-LLM in your research, please cite:

```bibtex
@article{you2025loftllm,
  title={LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models},
  author={You, Jiacheng and Yang, Jingcheng and Xie, Yuhang and Wu, Zhongxuan and Li, Xiucheng and Li, Feng and Wang, Pengjie and Xu, Jian and Zheng, Bo and Chen, Xinyang},
  journal={arXiv preprint arXiv:2512.20002},
  year={2025},
  url={https://arxiv.org/abs/2512.20002}
}
```

## Acknowledgements and license

This project builds on [Time-Series-Library](https://github.com/thuml/Time-Series-Library). See [LICENSE](LICENSE) for the MIT license and copyright notice.
