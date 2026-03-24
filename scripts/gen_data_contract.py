from pathlib import Path


def main():
    content = r"""# Data Contract for Stock Pattern ML

## 1. 目的

本文档用于约定模型训练阶段所需的数据输入格式，确保：

- 标注数据与行情数据能够正确对齐
- 数据集构建脚本能够稳定运行
- 不同同事产出的数据口径一致
- 后续模型训练、验证、推理流程可复用

当前任务聚焦于：

识别“横盘蓄势后放量启动”的第一阶段形态。

---

## 2. 数据交付范围

本项目当前需要两类输入数据：

1. 标签样本数据（labels）
2. 个股日线行情数据（daily parquet）

---

## 3. 标签数据规范

标签文件默认路径：

data/labels/labels.csv

### 3.1 必填字段

ts_code: 股票代码（如 000001.SZ）
asof_date: 样本日期（YYYY-MM-DD）
label: 标签（1=正样本，0=负样本）

### 3.2 推荐字段

sample_id: 样本唯一ID（ts_code + asof_date）
label_source: 标签来源（manual/rule/mock）
confidence: 置信度（0~1）

### 3.3 示例

sample_id,ts_code,asof_date,label,label_source,confidence
000001.SZ_2025-03-21,000001.SZ,2025-03-21,1,manual,0.95

### 3.4 标签语义

- asof_date 为观察终点
- 使用其之前的历史数据构建窗口
- label=1 表示符合形态
- label=0 表示不符合

### 3.5 注意事项

- 同一股票可多个样本
- 避免重复样本
- 无 sample_id 时自动生成

---

## 4. 日线数据规范

目录：

data/raw/daily/

文件名：

{ts_code}.parquet

### 4.1 必填字段

trade_date
open
high
low
close
vol

### 4.2 可选字段

amount
turnover_rate
pct_chg

### 4.3 数据要求

- trade_date 可解析
- 不重复
- 按时间排序
- 价格 > 0
- vol >= 0

---

## 5. 格式规范

股票代码：

000001.SZ / 600000.SH

日期格式：

YYYY-MM-DD

---

## 6. 窗口规则

- 截止 asof_date
- 向前取 window_size
- 不足 min_history 则丢弃

推荐：

window_size = 120
min_history = 160

---

## 7. 数据校验

标签检查：

- 必填字段
- label ∈ {0,1}
- 无重复

行情检查：

- 文件存在
- 字段齐全
- 日期合法

对齐检查：

- 文件匹配
- 时间覆盖
- 历史长度满足

---

## 8. 数据交付

labels:

data/labels/labels.csv

daily:

data/raw/daily/*.parquet

---

## 9. 非目标

当前不包含：

- 分钟数据
- 基本面
- 第二波标签
- 收益率标签

---

## 10. 版本管理

labels_v1.csv
labels_v2.csv

记录变更：

- 样本变化
- 标签规则变化

---

## 11. 最小样本

- 股票 ≥ 3
- 历史 ≥ 160天
- 标签 ≥ 6
- 正负都有
"""

    output_path = Path("docs/data_contract.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"data_contract.md generated at: {output_path}")


if __name__ == "__main__":
    main()