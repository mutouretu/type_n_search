# Dev Progress - Stock Pattern ML

## 📅 Date
2026-XX-XX

---

## 1. 当前阶段总结（Phase Summary）

本阶段已完成从 **数据 → 训练 → 推理 → 对比** 的完整闭环。

已具备能力：

- 数据协议定义完成（data_contract）
- 数据校验工具可用（check_data_contract.py）
- dataset 构建链路稳定（build_dataset.py）
- baseline 训练链路完成（LR / LGBM / XGB）
- 推理扫描链路完成（run_scan.py）
- 分析工具完成（inspect_dataset / inspect_predictions）
- 模型对比工具完成（compare_models.py）

👉 当前项目已进入：**可接真实数据阶段**

---

## 2. 数据层

### 数据协议（已固化）

- labels:
  - sample_id, ts_code, asof_date, label
- daily:
  - trade_date, open, high, low, close, vol

### 文件结构

data/
  raw/daily/{ts_code}.parquet
  labels/labels.csv
  processed/

---

## 3. 特征层

当前基础特征：

- ret_1d
- ma_5
- ma_20

统一入口：
src/features/indicators.py


说明：

- 训练 / 推理 / 扫描 已统一指标逻辑
- 避免多处重复实现

---

## 4. 模型层（Baseline）

当前已接入：

| Model | 状态 | 备注 |
|------|------|------|
| Logistic Regression | ✅ | 当前表现最佳 |
| LightGBM | ✅ | 非线性 baseline |
| XGBoost | ✅ | 非线性 baseline |

模型加载方式：

- 统一通过 `factory.load_model`
- 不再直接 pickle.load

---

## 5. 实验结果（Richer Mock）

### 数据集

- 样本数：240
- 正样本占比：~36%
- split：train / valid / test

---

### 模型表现（Validation）

| Model | Acc | F1 | AUC |
|------|-----|----|-----|
| LR   | ⭐ 最优 |
| LGBM | 次优 |
| XGB  | 第三 |

---

### 模型关系

- LR vs XGB：相关性高，但存在分歧
- LGBM vs XGB：高度一致（树模型族）
- Top-N 存在稳定交集（60%~90%）

---

## 6. 推理链路

已完成：

- TabularPredictor
- run_scan pipeline
- score 排序输出

输出文件：
outputs/predictions/scan_predictions*.csv


---

## 7. 工程能力评估

当前系统已具备：

- 可复现（config 驱动）
- 可扩展（model factory）
- 可验证（inspect / compare）
- 可对接真实数据

---

## 8. 当前限制

- 使用 mock 数据（非真实市场分布）
- 时间切分较简单（非 rolling）
- 特征仍较少
- 未做超参优化

---

## 9. 下一阶段计划（Phase 2）

优先级：

1. 接入真实数据
2. 验证数据质量（check_data_contract）
3. 跑三模型 baseline
4. 分析真实数据表现
5. 决定是否：
   - 调参（LGBM/XGB）
   - 扩特征
   - 上深度模型

---

## 10. 当前结论

- 三模型 baseline 已完整跑通
- pipeline 已工程化闭环
- richer mock 已验证系统有效性

👉 **项目已 ready for real data**