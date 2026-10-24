# LoFT-LLM

论文 **LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models** 的开源代码。
[论文](https://arxiv.org/abs/2512.20002v3) · [方法对应关系](docs/METHOD.md) · [改动记录](docs/CHANGES.md)

本次整理尽量保留原始实现：模型、注意力层、数据加载、指标、领域 prompt 和原运行入口保持原样；主要增加统一配置、启动器及缺失的 QLoRA 微调脚本。整理标准是完整提供论文提出的方法，不以复现论文表格为验收条件。

## 安装与数据

建议使用 Python 3.10 和 CUDA 环境，按本机 CUDA 配置安装合适的 PyTorch，再安装依赖：

```bash
pip install -r requirements.txt
```

`requirements-original.txt` 保留原始依赖清单。当前清单只更新了旧版 PyTorch 约束并补充 Qwen3/QLoRA 依赖，未声称是原实验环境锁定文件。

数据格式沿用原始代码，见 [docs/DATA.md](docs/DATA.md)。配置分别为 `configs/fundar.json`、`configs/solar.json`。修改其中的数据目录、`cuda_visible_devices`、基础模型路径、适配器路径及预测长度即可，无需编辑算法代码。基础模型可以是 `Qwen/Qwen3-8B` 或本地模型目录。

## 原始三阶段流程

所有命令在本目录运行。将 `fundar` 换成 `solar` 即使用 Solar 配置。

```bash
# 1. PLFM 低频学习：原 run_plfm.py -> Exp_Frequency
python run_release.py --config configs/fundar.json --stage plfm

# 2. 固定 PLFM，训练原 iTransformer 残差学习器
python run_release.py --config configs/fundar.json --stage residual

# 3. 原代码生成 prompt + 真实数值列表，随后 QLoRA 微调 Qwen3-8B
python run_release.py --config configs/fundar.json --stage sft
python run_release.py --config configs/fundar.json --stage qlora

# 加载微调适配器，运行原始 LLM 校准与评估入口
python run_release.py --config configs/fundar.json --stage infer
```

每条命令支持 `--horizon 3`，同一流程的各阶段使用相同预测长度。监督样本和适配器分别保存在 `outputs/<领域>/h<长度>/`、`adapters/<领域>/h<长度>/`；PLFM/残差检查点仍使用原始命名格式。

先查看实际调用及参数，无需安装模型依赖：

```bash
python run_release.py --config configs/solar.json --stage infer --horizon 3 --dry-run
```

也可继续直接使用原 `run_plfm.py`、`run_TSLFL.py` 和 `run_TSLFL_LLM_data.py`。金融和 Solar 的 LLM 路径会分别读取对应默认配置；使用自定义配置时建议通过 `run_release.py` 启动。

## 检查

```bash
python -m unittest discover -s tests -v
python train_qlora.py --config configs/fundar.json --check-data
```

第一条检查配置是否兼容原参数解析器，以及新增 SFT 脚本的监督标签。第二条在监督样本导出后检查原 JSON 数据格式，不加载大模型。当前整理环境未安装 PyTorch、没有 GPU 权重，因此未执行训练和大模型推理；已执行检查见 [docs/VALIDATION.md](docs/VALIDATION.md)。

原项目中的 TSLib 基线脚本、TS-LFL 历史目录等保留，供已有用法兼容；论文主方法使用上面的配置与入口。原 TSLib 说明另存为 `README_TSLib.md`，原许可证保留在 `LICENSE`。
