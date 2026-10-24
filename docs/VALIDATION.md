# 已执行检查与范围

本次目标是尽量少改原代码、完整提供论文方法，不复现论文数值结果。

已执行：136 个 Python 文件语法解析通过；8 项配置/监督标签检查全部通过，其中覆盖两个领域、预测长度 1/3/10、四个原运行阶段的参数解析兼容；检查点命名与领域适配器检查；新增微调脚本的数据格式、回复监督和 EOS/填充处理检查；原目录与整理目录的逐文件比较。

`models/`、`layers/`、`data_provider/`、`utils/` 及原 `run*.py` 入口保持逐字节一致。原始 419 个文件的保留/改动详情在 `FILE_CHANGES.json`；原始 Python 内容差异在 `docs/original_code_changes.diff`。

当前环境没有 PyTorch、Transformers/PEFT 和模型权重，没有执行原模型训练、GPU QLoRA 或大模型推理。这不改变本次按方法一致性整理的任务范围，也不将论文数字复现作为发布前提。

在准备好依赖和数据后，可按 README 依次运行原始三阶段流程。新增 QLoRA 接口依据 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)、[PEFT 量化文档](https://huggingface.co/docs/peft/developer_guides/quantization) 和 [Transformers Trainer](https://huggingface.co/docs/transformers/v4.51.3/main_classes/trainer) 实现。
