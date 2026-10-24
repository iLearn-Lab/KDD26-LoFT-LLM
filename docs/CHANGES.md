# 最小改动说明

整理基于本地 `LLM_project/`，不是重新实现论文。原始 419 个文件均保留；逐文件比较在 [FILE_CHANGES.json](../FILE_CHANGES.json)。

## 原始 Python 文件仅改五个

1. **`exp/exp_TSLFL_LLM.py`**：选回原 `exp_TSLFL_LLM beifen.py` 的金融 Qwen3-8B 实现，替代实验性的 LLaMA 入口；此后仅把基础模型、适配器及检查点位置改为配置读取。
2. **`exp/exp_TSLFL_LLM_Solar.py`**：基础模型、Solar 适配器和检查点路径配置化；移除 `print(None.shape)` 调试异常。
3. **`exp/exp_TSLFL_LLM_train_data.py`**：去掉仅针对私有第 9/10 步检查点的硬编码，按当前配置长度和原命名规则选检查点；监督 JSON 输出路径配置化。prompt、真实值构造和抽样比例不改。
4. **`exp/exp_TSLFL_LLM_Solar_train_data.py`**：同上，不再固定遍历原第 1–8 步私有检查点；样本内容与生成逻辑不改。
5. **`exp/exp_frequency.py`**：两处 `irfft` 增加 `n=self.pred_len`，仅恢复奇数长度目标。模型和频域损失表达式不改。

金融入口相对于所选原备份的差异见 [financial_backup_changes.diff](financial_backup_changes.diff)；相对于整个 main 原目录的完整代码差异见 [original_code_changes.diff](original_code_changes.diff)。

`models/`、`layers/`、`data_provider/`、`utils/`、原 `run*.py` 文件逐字节保持一致。没有采用重构版模型，没有合入 temp 的小模型实验、不同数据加载逻辑或指标改动。

## 新增与文档配置

- `release_config.py`、`run_release.py`：JSON 配置、原命令行参数转换与启动。保持原入口兼容。
- `configs/fundar.json`、`configs/solar.json`：数据位置、GPU、模型、领域适配器、预测长度和各阶段参数。
- `train_qlora.py`：补齐论文第三阶段，直接读取原监督样本 JSON，不改动原导出格式。超参数在配置中公开。
- `tests/test_release.py`：检查新配置与原参数解析器兼容、检查点命名和新增 SFT 标签处理。
- README、数据说明、方法对应及检查记录。原 TSLib README 和依赖列表分别另存。
- `requirements.txt` 仅替换旧 Torch 约束并添加 Qwen/QLoRA 所需四个依赖；`.gitignore` 增加训练产物目录。

本次不统一原数据拆分、不改变缩放、prompt 领域规则、指标或训练损失，不以历史实验数字是否能重现作为代码整理的前提。
