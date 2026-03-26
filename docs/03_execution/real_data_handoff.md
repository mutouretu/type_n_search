
# Real Data Handoff（P0 数据对接说明）

## 1. 本轮目标

本轮数据交付的目标：

> 验证真实数据是否满足数据协议（data_contract.md），并能够跑通完整 pipeline：

```text
check_data_contract
→ build_dataset
→ train_model
→ run_scan
```

本阶段不关注模型效果，仅关注：

* 数据是否符合协议
* 是否存在结构性问题
* 是否能够稳定进入模型系统

---

## 2. 协议版本

本轮数据交付基于以下协议：

```text
docs/data_contract.md (当前版本)
```

所有数据必须满足该协议定义的字段、语义及约束。

---

## 3. 本轮交付范围

### 3.1 样本规模

建议范围：

```text
50 ~ 200 个样本
```

目的：

* 覆盖多种情况
* 验证系统稳定性
* 不追求规模

---

### 3.2 样本类型（必须覆盖）

本轮数据必须包含：

* 正样本（label=1）
* 负样本（label=0）
* 边界样本（形态不明显）
* 历史不足样本（测试过滤逻辑）
* 异常样本（缺字段 / 数据问题）

---

## 4. 数据路径约定

本轮使用独立目录：

```text
data/raw_real/
  ├── labels/
  │   └── labels.csv
  └── daily/
      ├── <ts_code>.parquet
      └── ...
```

说明：

* 不与 mock 数据混用
* 每个股票一个 parquet 文件
* daily 数据需覆盖 labels 中所有 ts_code

---

## 5. 数据要求（关键约束）

### 5.1 Labels

必须字段：

```text
sample_id
ts_code
asof_date
label
```

约束：

1. sample_id 必须唯一（由数据侧生成并固定）
2. asof_date 表示“判断时点”
3. label 不得使用未来信息（禁止信息泄露）

---

### 5.2 Daily

必须字段：

```text
trade_date
open
high
low
close
vol
```

约束：

1. 按 trade_date 升序
2. 每个 ts_code 内 trade_date 唯一
3. 覆盖 asof_date 之前的历史数据
4. 至少满足窗口长度要求

---

### 5.3 对齐规则

对于每个样本：

```text
(ts_code, asof_date)
```

必须满足：

* daily 数据中存在该股票
* 可回溯窗口数据
* 不允许未来数据参与构建

---

## 6. 验收流程

数据交付后，由模型侧执行：

```bash
python -m src.pipelines.check_real_data
```

然后执行：

```bash
python -m src.pipelines.build_dataset
python -m src.pipelines.train_model
python -m src.pipelines.run_scan
```

---

## 7. 验收产物

必须生成：

```text
outputs/analysis/real_data_quality_report.md
outputs/processed_data/
outputs/models/
outputs/predictions/
```

---

## 8. 判定标准

### 必须通过

* schema 正确
* sample_id 唯一
* 时间无错位
* dataset 构建成功
* 模型训练成功
* scan 可输出结果

---

### 允许警告

* 少量数据缺失
* 个别样本被过滤

---

### 判定失败

以下情况视为不通过：

* 无法构建 dataset
* label 与数据严重不匹配
* 大规模样本被过滤
* 时间存在明显错误

---

## 9. 问题反馈机制

模型侧将输出问题报告，包括：

* 字段缺失
* 时间错位
* 数据不完整
* 样本异常

数据侧根据报告修复后重新交付。

---

## 10. 本轮完成标志

当满足以下条件时，本轮对接完成：

* 数据通过 contract 校验
* pipeline 全流程可运行
* 无阻塞性数据问题

---

## 11. 注意事项（非常重要）

1. 本阶段不优化模型效果
2. 优先保证数据正确性
3. 所有问题必须可复现
4. 严禁使用未来信息

---

## 12. 下一阶段（Phase 2 - P1）

本轮完成后，将进入：

* 评估体系升级
* 特征扩展
* 模型对比分析

---
