# Type N Search

量化选股机器学习最小工程模板。当前版本聚焦于：
- 基于日线窗口构建样本
- 进行数据协议校验（data contract）
- 产出 tabular 训练数据
- 训练一个可跑通的 baseline（Logistic Regression）

## 项目目标

- 统一样本协议（`SampleMeta` + `sample_id`）
- 在正式构建训练集前做数据校验，尽早拦截脏数据
- 提供最小可运行流水线：`mock data -> check -> build dataset -> train`
- 为后续接入更复杂模型（序列模型、多因子特征）保留接口

## 目录结构

```text
type_n_search/
├── configs/
│   ├── data.yaml
│   └── train.yaml
├── data/
│   ├── labels/
│   │   └── labels.csv
│   ├── raw/
│   │   └── daily/                 # 原始日线 parquet
│   └── processed/                 # dataset build 产物
├── docs/
│   └── data_contract.md
├── scripts/
│   ├── generate_mock_data.py      # 生成 mock 日线
│   └── check_data_contract.py     # 数据协议校验
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   │   ├── build_dataset.py
│   │   └── train_model.py
│   └── training/
└── requirements.txt
```

## 如何生成 Mock 数据

```bash
python scripts/generate_mock_data.py
```

默认生成 3 只股票（约 220 个交易日）的日线 parquet 到 `data/raw/daily/`：
- `000001.SZ`
- `000002.SZ`
- `600000.SH`

## 如何验数

```bash
python scripts/check_data_contract.py \
  --labels-path data/labels/labels.csv \
  --raw-daily-dir data/raw/daily \
  --min-history 160
```

校验内容包括：
- labels 必需列/值合法性
- 日线文件存在性与列完整性
- `(ts_code, asof_date)` 对齐与历史长度
- `sample_id == {ts_code}_{asof_date}` 强一致性

## 如何 Build Dataset

```bash
python src/pipelines/build_dataset.py --config configs/data.yaml
```

执行顺序：
1. 先跑 data contract 校验（失败即退出）
2. 再构建训练集产物

默认产物在 `data/processed/`：
- `sample_meta.parquet`
- `X_tabular.parquet`
- `y.npy`
- `X_sequence.npy`（可选）

## 如何训练 Baseline

```bash
python src/pipelines/train_model.py --config configs/train.yaml
```

当前 baseline：
- `model_name: logistic_regression`
- 训练器输出：
  - `model.pkl`
  - `metrics.json`
  - `valid_predictions.csv`

## 真实数据交付格式

### 1) 日线数据（按股票分文件）
- 路径：`data/raw/daily/{ts_code}.parquet`
- 必需列：
  - `trade_date`
  - `open`
  - `high`
  - `low`
  - `close`
  - `vol`
- 可选列：
  - `amount`
  - `turnover_rate`
  - `pct_chg`

### 2) 标签数据
- 路径：`data/labels/labels.csv`
- 推荐列：
  - `sample_id`
  - `ts_code`
  - `asof_date`
  - `label`
  - `label_source`
  - `confidence`

说明：
- `sample_id` 建议严格使用 `{ts_code}_{YYYY-MM-DD}`
- `label` 当前按二分类处理（0/1）
