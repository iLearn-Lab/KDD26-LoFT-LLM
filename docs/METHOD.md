# 论文方法与原始代码对应

对应本地论文 / [arXiv:2512.20002v3](https://arxiv.org/html/2512.20002v3)。本次不重写算法，以提供完整方法代码和正确配置为整理目标。

| 论文部分 | 原始实现 | 整理处理 |
| --- | --- | --- |
| §3.1 FALoss | `exp/exp_frequency.py`、`exp/exp_TSLFL.py` 中的频谱绝对误差 | 损失表达式保持原样 |
| §3.2 PLFM、重叠 patch、实虚部双路 MLP | `models/PLFM.py` | 文件逐字节保持原样 |
| §3.2 低通目标监督 | `exp/exp_frequency.py` | 仅给两处 IFFT 增加原预测长度参数，避免奇数窗口被缩短 |
| §3.3 iTransformer 残差 | `models/iTransformer_TSLFL.py` | 高频输入处理、骨干、低频/残差输出保持原样 |
| §3.3 固定 PLFM 训练残差 | `exp/exp_TSLFL.py` | 文件保持原样；原优化器只更新残差预测器参数 |
| §3.4 金融 Qwen3-8B 校准 | 原 `exp_TSLFL_LLM beifen.py`，恢复为 `exp/exp_TSLFL_LLM.py` | 选择原来的 Qwen 实现，基础模型与金融适配器路径改为配置读取 |
| §3.4 Solar 校准 | `exp/exp_TSLFL_LLM_Solar.py` | 原 Qwen/Solar 路线；路径配置化，移除调试异常 |
| §3.4 / 附录 A 领域 prompt | `layers/GetFinancialPrompt.py`、`GetSolarPrompt.py` 及对应处理工具 | 文件逐字节保持原样 |
| 附录 A.3 监督样本 | `exp/exp_TSLFL_LLM_train_data.py`、`exp_TSLFL_LLM_Solar_train_data.py` | 原 prompt/真实值生成过程不变，只改检查点选择和输出位置 |
| §3.4 QLoRA / SFT | 新增 `train_qlora.py` | 读取原 messages JSON，NF4 量化 + LoRA，仅对回复和 EOS 监督 |

`run_release.py` 只把 JSON 配置转为原运行入口的命令行参数，不替换模型或数据加载流程。LoRA rank、学习率等新增脚本默认值公开在配置中，不声明还原了历史微调超参数。论文中的数值表格无需重新训练验证才能完成本次代码整理。
