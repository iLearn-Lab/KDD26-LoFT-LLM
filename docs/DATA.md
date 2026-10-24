# 原数据格式与路径配置

本次保留 `data_provider/` 全部实现，不改变数据处理、拆分、缩放或列顺序。请使用与原代码相同的预处理输入，数据目录在 JSON 配置的 `common.root_path` 中设置。

## FundAR

- `ProductAmt.csv`：`date` 加 88 个目标数值列，供 PLFM 和残差阶段使用。
- `ProductAmt_with_info.csv`：`date` 后首先为同样顺序的 88 个目标列，其后为领域辅助信息。
- 原实现按 `apply_amt_<基金ID>`、`redeem_amt_<基金ID>` 以及其他带基金后缀的字段分组。保留其 44 个基金 / 88 个目标通道设定。
- 原 prompt 识别 `yield_`、`uv_fundown`、`uv_stableown`、`uv_fundopt`、`uv_fundmarket`、`is_trade`、`is_month_end` 等字段；列名与顺序应与原处理函数一致。
- 数据来源见论文所用的 [天池基金流量预测比赛](https://tianchi.aliyun.com/competition/entrance/532224/information)。本库不附带原始比赛数据。

## Solar

- `Solar.csv`：`date` 加一个功率目标列，供前两个阶段使用。
- `Solar_Info.csv`：`date` 后第一个数值列为目标，随后是原气象辅助列；整体至少 13 个数值特征。
- `layers/SolarLLMProcess_utils.py::compute_derived_features_tensor` 按位置读取特征。以目标为数值列位置 0，已使用的位置为：1=`tclw`，2=`tciw`，4=`r`，5=`TCC`，6=`u10`，7=`v10`，9=`SSRD`，12=`TP`。其他位置也应按原数据排列保留。
- 原 Solar prompt 包含当地时间和归一化条件，使用自己的站点数据时需对应其时区、量纲及列格式；本次未重写这些领域规则。

两领域的 `_LLM_Train` 加载类只取训练区间供监督样本生成。未来辅助变量的使用沿用论文假设，实际推理时应提供预测时可获得的辅助信息。PLFM 的 `forward` 虽保留原多参数接口，但其预测只使用历史序列，不使用传入的未来目标。
