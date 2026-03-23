#!/usr/bin/env bash

set -e

PROJECT_NAME="stock_pattern_ml"

echo "Creating project scaffold: ${PROJECT_NAME}"

mkdir -p "${PROJECT_NAME}"

mkdir -p "${PROJECT_NAME}/configs"

mkdir -p "${PROJECT_NAME}/data/raw/daily"
mkdir -p "${PROJECT_NAME}/data/labels"
mkdir -p "${PROJECT_NAME}/data/interim"
mkdir -p "${PROJECT_NAME}/data/processed"

mkdir -p "${PROJECT_NAME}/notebooks"

mkdir -p "${PROJECT_NAME}/src/common"
mkdir -p "${PROJECT_NAME}/src/data"
mkdir -p "${PROJECT_NAME}/src/features"
mkdir -p "${PROJECT_NAME}/src/labeling"
mkdir -p "${PROJECT_NAME}/src/models"
mkdir -p "${PROJECT_NAME}/src/models/tabular"
mkdir -p "${PROJECT_NAME}/src/models/sequence"
mkdir -p "${PROJECT_NAME}/src/training"
mkdir -p "${PROJECT_NAME}/src/inference"
mkdir -p "${PROJECT_NAME}/src/pipelines"

mkdir -p "${PROJECT_NAME}/outputs/models"
mkdir -p "${PROJECT_NAME}/outputs/metrics"
mkdir -p "${PROJECT_NAME}/outputs/predictions"
mkdir -p "${PROJECT_NAME}/outputs/logs"

mkdir -p "${PROJECT_NAME}/tests"

touch "${PROJECT_NAME}/README.md"
touch "${PROJECT_NAME}/requirements.txt"

touch "${PROJECT_NAME}/configs/data.yaml"
touch "${PROJECT_NAME}/configs/feature.yaml"
touch "${PROJECT_NAME}/configs/train.yaml"
touch "${PROJECT_NAME}/configs/infer.yaml"

touch "${PROJECT_NAME}/src/__init__.py"

touch "${PROJECT_NAME}/src/common/__init__.py"
touch "${PROJECT_NAME}/src/common/paths.py"
touch "${PROJECT_NAME}/src/common/logger.py"
touch "${PROJECT_NAME}/src/common/utils.py"
touch "${PROJECT_NAME}/src/common/seed.py"

touch "${PROJECT_NAME}/src/data/__init__.py"
touch "${PROJECT_NAME}/src/data/schema.py"
touch "${PROJECT_NAME}/src/data/loader.py"
touch "${PROJECT_NAME}/src/data/validator.py"
touch "${PROJECT_NAME}/src/data/splitter.py"
touch "${PROJECT_NAME}/src/data/dataset_builder.py"

touch "${PROJECT_NAME}/src/features/__init__.py"
touch "${PROJECT_NAME}/src/features/indicators.py"
touch "${PROJECT_NAME}/src/features/window_builder.py"
touch "${PROJECT_NAME}/src/features/feature_builder_tabular.py"
touch "${PROJECT_NAME}/src/features/feature_builder_sequence.py"
touch "${PROJECT_NAME}/src/features/normalizer.py"

touch "${PROJECT_NAME}/src/labeling/__init__.py"
touch "${PROJECT_NAME}/src/labeling/label_schema.py"
touch "${PROJECT_NAME}/src/labeling/sample_registry.py"
touch "${PROJECT_NAME}/src/labeling/hard_negative_mining.py"

touch "${PROJECT_NAME}/src/models/__init__.py"
touch "${PROJECT_NAME}/src/models/base.py"
touch "${PROJECT_NAME}/src/models/factory.py"

touch "${PROJECT_NAME}/src/models/tabular/__init__.py"
touch "${PROJECT_NAME}/src/models/tabular/baseline_stub.py"

touch "${PROJECT_NAME}/src/models/sequence/__init__.py"
touch "${PROJECT_NAME}/src/models/sequence/baseline_stub.py"

touch "${PROJECT_NAME}/src/training/__init__.py"
touch "${PROJECT_NAME}/src/training/trainer.py"
touch "${PROJECT_NAME}/src/training/evaluator.py"
touch "${PROJECT_NAME}/src/training/metrics.py"
touch "${PROJECT_NAME}/src/training/callbacks.py"

touch "${PROJECT_NAME}/src/inference/__init__.py"
touch "${PROJECT_NAME}/src/inference/predictor.py"
touch "${PROJECT_NAME}/src/inference/scanner.py"
touch "${PROJECT_NAME}/src/inference/postprocess.py"

touch "${PROJECT_NAME}/src/pipelines/__init__.py"
touch "${PROJECT_NAME}/src/pipelines/build_dataset.py"
touch "${PROJECT_NAME}/src/pipelines/train_model.py"
touch "${PROJECT_NAME}/src/pipelines/validate_model.py"
touch "${PROJECT_NAME}/src/pipelines/run_scan.py"

touch "${PROJECT_NAME}/tests/test_features.py"
touch "${PROJECT_NAME}/tests/test_dataset_builder.py"
touch "${PROJECT_NAME}/tests/test_splitter.py"

echo "Done."
echo "Project created at: ${PROJECT_NAME}"