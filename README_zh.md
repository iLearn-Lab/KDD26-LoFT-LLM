# LoFT-LLM

论文 **LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models** 的开源代码。
[论文](https://arxiv.org/abs/2512.20002v3) · [English](README.md)

LoFT-LLM 将低频预测、残差学习与领域大模型校准结合起来。训练包含三个阶段：PLFM 低频学习、固定 PLFM 后的 iTransformer 残差学习，以及 Qwen3-8B 的 QLoRA 监督微调。

## 安装

建议使用 Python 3.10 和 CUDA 环境，按本机 CUDA 配置安装合适的 PyTorch，再安装依赖：

```bash
pip install -r requirements.txt
```

## 数据与配置

金融和 Solar 的配置分别为 `configs/fundar.json`、`configs/solar.json`；在其中设置数据目录、`cuda_visible_devices`、预测长度及模型路径。基础模型可以是 [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) 或本地模型目录。数据和训练后的适配器需要另行获取或生成。

`common.root_path` 指定数据目录，`stages.<阶段>.data_path` 指定对应 CSV 文件名。数据加载器要求以下格式：

| 数据集 | 文件 | 列格式 |
| --- | --- | --- |
| FundAR | `ProductAmt.csv` | `date` 后接 88 个数值目标列 |
| FundAR | `ProductAmt_with_info.csv` | `date` 后先放相同顺序的 88 个目标列，再放辅助信息 |
| Solar | `Solar.csv` | `date` 后接一个功率目标列 |
| Solar | `Solar_Info.csv` | `date` 后先放功率目标，再放气象特征；总计至少 13 个数值列 |

FundAR 使用 44 个基金，以 `apply_amt_<基金ID>`、`redeem_amt_<基金ID>` 为目标列。辅助字段包含 `yield_`、`uv_fundown`、`uv_stableown`、`uv_fundopt`、`uv_fundmarket`、`is_trade` 和 `is_month_end` 等，列名与顺序须匹配 `layers/FinancialLLMProcess_utils.py`。数据来源为[天池基金流量预测比赛](https://tianchi.aliyun.com/competition/entrance/532224/information)。

Solar 特征由 `layers/SolarLLMProcess_utils.py` 按位置读取。以目标为数值列位置 0，使用的位置包括：1=`tclw`、2=`tciw`、4=`r`、5=`TCC`、6=`u10`、7=`v10`、9=`SSRD`、12=`TP`。须匹配该函数要求的完整列序，以及 prompt 中的时区、量纲与归一化方式。

监督样本仅从训练区间生成。推理时应提供预测时可获得的辅助信息。

## 训练与推理

在本目录依次执行各阶段。将 `fundar` 换成 `solar` 即使用 Solar 配置。

```bash
# 1. PLFM 低频学习
python run_release.py --config configs/fundar.json --stage plfm

# 2. 固定 PLFM，训练 iTransformer 残差学习器
python run_release.py --config configs/fundar.json --stage residual

# 3. 导出监督样本，随后 QLoRA 微调 Qwen3-8B
python run_release.py --config configs/fundar.json --stage sft
python run_release.py --config configs/fundar.json --stage qlora

# 加载微调适配器，运行 LLM 校准与评估
python run_release.py --config configs/fundar.json --stage infer
```

每条命令支持 `--horizon 3`，同一流程的各阶段使用相同预测长度。监督样本和适配器分别保存在 `outputs/<领域>/h<长度>/`、`adapters/<领域>/h<长度>/`。检查点路径及 QLoRA 超参数在对应领域的 JSON 文件中配置。

使用 `--dry-run` 可以在不加载模型的情况下查看运行命令及参数：

```bash
python run_release.py --config configs/solar.json --stage infer --horizon 3 --dry-run
```

## 检查

```bash
python -m unittest discover -s tests -v
python train_qlora.py --config configs/fundar.json --check-data
```

第一条运行配置与监督标签的单元检查；第二条在监督样本导出后检查 JSON 数据格式，无需加载大模型。

## 致谢与许可

本项目基于 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 构建。MIT 许可证和版权声明见 [LICENSE](LICENSE)。
