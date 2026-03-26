# Phase 2 开发路线图（Real Data & Evaluation Stage）

## 1. 背景与目标

当前项目已完成 Phase 1：

* 数据协议（data contract）已定义
* mock 数据链路已跑通
* Dataset 构建、训练、扫描、分析闭环已建立
* 三模型 baseline（LR / LGBM / XGB）可稳定运行

但当前系统仍存在以下限制：

* 使用 mock 数据（非真实市场）
* 时间切分为单次切分（非 rolling）
* 特征数量较少
* 未进行系统性调参与评估优化

---

## Phase 2 核心目标：

> 将系统从“可运行”升级为“可基于真实数据进行策略判断”。

---

## 2. 总体策略

Phase 2 按以下顺序推进：

```
P0 → 真实数据准入与可复现
P1 → 评估体系升级 + 特征扩展
P2 → 稳定性验证（rolling）+ 调参与模型决策
```

⚠️ 原则：

* 不优先追求复杂模型
* 优先保证数据质量与评估可靠性

---

## 3. P0：真实数据准入关（必须优先完成）

### 3.1 构建最小真实数据样本集

目标：验证系统是否能正确处理真实数据

覆盖样本类型：

* 正样本
* 负样本
* 边界样本
* 历史不足样本
* 异常/缺失数据样本

目录建议：

```
data/raw_real/
  ├── labels/
  │   └── labels.csv
  └── daily/
      ├── 000001.SZ.parquet
      └── ...
```

---

### 3.2 数据质量校验升级

在现有 `check_data_contract` 基础上增加：

#### 校验项：

1. 主键唯一性
2. 时间合法性（无未来数据）
3. 历史长度充足
4. 标签分布合理性
5. 日线数据质量（排序、缺失、异常）

#### 新增模块：

```
src/data/quality_report.py
src/pipelines/check_real_data.py
```

#### 输出：

```
outputs/analysis/real_data_quality_report.json
outputs/analysis/real_data_quality_report.md
```

---

### 3.3 真实数据 Smoke Pipeline

目标：验证全链路可运行

流程：

```
check_data_contract
→ build_dataset
→ train_model (LR / LGBM / XGB)
→ run_scan
→ inspect / compare
```

建议新增：

```
scripts/run_phase2_smoke.sh
```

---

### 3.4 配置隔离（Mock vs Real）

新增：

```
configs/data/real_data.yaml
configs/train/lr_real.yaml
configs/train/lgbm_real.yaml
configs/train/xgb_real.yaml
configs/scan/real_scan.yaml
```

---

### 3.5 端到端测试

新增测试：

```
tests/functional/test_phase2_realdata_smoke.py
```

验证：

* contract check
* dataset 构建
* 模型训练
* scan 输出

---

## 4. P1：评估体系升级

### 4.1 引入选股导向指标

当前仅有：

* AUC / F1 / Acc

需要新增：

* Top-K Precision
* Top-K 命中率
* Score 分桶命中率
* 分桶未来收益
* Rank-IC / IC
* 模型 Top-N 交集稳定性

---

### 4.2 新增分析模块

```
src/analysis/evaluate_selection.py
src/pipelines/evaluate_models.py
```

输出：

```
outputs/analysis/model_selection_report.csv
outputs/analysis/model_bucket_report.csv
```

---

### 4.3 分类能力 vs 选股能力分离

明确区分：

| 类型   | 指标              |
| ---- | --------------- |
| 分类能力 | AUC / F1        |
| 选股能力 | Top-K / 收益 / IC |

---

### 4.4 人工复核样本机制

新增：

```
data/review/review_candidates.csv
docs/review_samples.md
```

样本类型：

* 高分但异常样本
* 低分但疑似正样本
* 模型分歧样本

---

## 5. P1：特征扩展（围绕标签）

### 5.1 第一批特征（优先）

#### 横盘压缩类

* 振幅
* 波动率
* 均线粘合度

#### 启动确认类

* 放量倍数
* 突破前高
* 相对位置

#### 过滤类

* 成交额
* 停牌过滤
* 涨跌停过滤

---

### 5.2 落地位置

```
src/features/indicators.py
src/features/feature_builder.py
```

---

### 5.3 特征版本管理

在 `model_meta.json` 中记录：

```
feature_set: basic_v1 / breakout_v1 / full_v1
```

---

## 6. P2：时间验证升级

### 6.1 Rolling / Walk-forward

新增：

```
src/data/splitter.py（扩展）
```

支持：

* rolling split
* walk-forward split

---

### 6.2 多折评估输出

输出：

* mean / std 指标
* 各折表现
* Top-N 稳定性

---

## 7. P2：调参与模型策略

### 7.1 调参顺序

1. 固定数据
2. 固定特征
3. 调参

---

### 7.2 优先模型

* LGBM（优先）
* XGB
* LR（baseline）

---

### 7.3 是否上深度模型（门槛）

仅当满足：

* 标签稳定
* rolling 验证稳定
* 特征已充分扩展

---

## 8. 测试与工程保障

新增：

```
tests/functional/test_contract_and_build_real.py
tests/functional/test_train_and_scan_real.py
```

---

## 9. 里程碑定义

### Milestone A：真实数据接入完成

* dataset 构建成功
* 模型训练成功
* scan 可输出

---

### Milestone B：评估闭环完成

* 有选股指标
* 有模型对比
* 可判断优劣

---

### Milestone C：稳定性验证完成

* rolling 可运行
* 多折稳定
* 明确下一步方向

---

## 10. 最关键成功要素

Phase 2 成败取决于：

1. 数据质量
2. 标签一致性
3. 评估是否贴近交易
4. 时间验证是否可靠

---

## 11. 总结

Phase 2 的核心不是“更强模型”，而是：

> 建立一个可以用真实数据评估策略有效性的系统。

---
