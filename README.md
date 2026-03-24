# Type N Search

量化选股机器学习工程模板（当前已打通三模型 baseline）：
- `logistic_regression`
- `lightgbm`
- `xgboost`

支持完整链路：
`mock data -> data contract check -> build dataset -> train -> scan -> inspect -> compare`

## 项目目标

- 统一样本协议（`SampleMeta` + `sample_id`）
- 构建前先做数据协议校验，尽早拦截脏数据
- 提供可运行的 tabular baseline 训练/推理流程
- 支持多模型统一 save/load 与对比分析

## 目录结构

```text
type_n_search/
├── configs/
│   ├── data.yaml
│   ├── train.yaml
│   ├── train_lgbm.yaml
│   ├── train_xgb.yaml
│   ├── infer.yaml
│   ├── infer_lgbm.yaml
│   └── infer_xgb.yaml
├── data/
│   ├── labels/labels.csv
│   ├── raw/daily/                  # 按股票分 parquet
│   └── processed/                  # build_dataset 产物
├── docs/data_contract.md
├── outputs/
│   ├── models/
│   ├── predictions/
│   └── analysis/
├── scripts/
│   ├── generate_mock_data.py
│   ├── generate_richer_mock_data.py
│   ├── check_data_contract.py
│   ├── inspect_dataset.py
│   ├── inspect_predictions.py
│   └── compare_models.py
├── src/
│   ├── data/
│   ├── features/
│   ├── inference/
│   ├── models/
│   ├── pipelines/
│   └── training/
└── tests/
```

## 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 一键跑 richer mock 实验

```bash
python scripts/generate_richer_mock_data.py
python scripts/check_data_contract.py
python src/pipelines/build_dataset.py --config configs/data.yaml
python scripts/inspect_dataset.py --output-json outputs/analysis/dataset_inspection_rich.json --top-n 10

python src/pipelines/train_model.py --config configs/train.yaml
python scripts/inspect_predictions.py --pred-path outputs/models/baseline_lr/valid_predictions.csv --output-json outputs/analysis/lr_valid_inspection_rich.json --top-n 10

python src/pipelines/train_model.py --config configs/train_lgbm.yaml
python scripts/inspect_predictions.py --pred-path outputs/models/baseline_lgbm/valid_predictions.csv --output-json outputs/analysis/lgbm_valid_inspection_rich.json --top-n 10

python src/pipelines/train_model.py --config configs/train_xgb.yaml
python scripts/inspect_predictions.py --pred-path outputs/models/baseline_xgb/valid_predictions.csv --output-json outputs/analysis/xgb_valid_inspection_rich.json --top-n 10

python src/pipelines/run_scan.py --config configs/infer.yaml
python src/pipelines/run_scan.py --config configs/infer_lgbm.yaml
python src/pipelines/run_scan.py --config configs/infer_xgb.yaml
```

## 模型对比（验证/扫描）

```bash
python scripts/compare_models.py \
  --pred-a-path outputs/models/baseline_lr/valid_predictions.csv \
  --pred-b-path outputs/models/baseline_lgbm/valid_predictions.csv \
  --name-a lr --name-b lgbm --top-n 10 \
  --output-json outputs/analysis/compare_valid_lr_vs_lgbm_rich.json

python scripts/compare_models.py \
  --pred-a-path outputs/models/baseline_lr/valid_predictions.csv \
  --pred-b-path outputs/models/baseline_xgb/valid_predictions.csv \
  --name-a lr --name-b xgb --top-n 10 \
  --output-json outputs/analysis/compare_valid_lr_vs_xgb_rich.json

python scripts/compare_models.py \
  --pred-a-path outputs/models/baseline_lgbm/valid_predictions.csv \
  --pred-b-path outputs/models/baseline_xgb/valid_predictions.csv \
  --name-a lgbm --name-b xgb --top-n 10 \
  --output-json outputs/analysis/compare_valid_lgbm_vs_xgb_rich.json
```

## 真实数据交付格式

### 日线数据
- 路径：`data/raw/daily/{ts_code}.parquet`
- 必需列：`trade_date, open, high, low, close, vol`
- 可选列：`amount, turnover_rate, pct_chg`

### 标签数据
- 路径：`data/labels/labels.csv`
- 推荐列：`sample_id, ts_code, asof_date, label, label_source, confidence`

约定：
- `sample_id` 格式：`{ts_code}_{YYYY-MM-DD}`
- `label` 为二分类 `0/1`

## 产物说明

### build_dataset 输出
- `data/processed/sample_meta.parquet`
- `data/processed/X_tabular.parquet`
- `data/processed/y.npy`
- `data/processed/X_sequence.npy`（可选）

### train_model 输出
- `outputs/models/<model_name>/model.pkl`
- `outputs/models/<model_name>/model_meta.json`
- `outputs/models/<model_name>/normalizer.pkl`
- `outputs/models/<model_name>/metrics.json`
- `outputs/models/<model_name>/valid_predictions.csv`

### run_scan 输出
- `outputs/predictions/scan_predictions*.csv`
