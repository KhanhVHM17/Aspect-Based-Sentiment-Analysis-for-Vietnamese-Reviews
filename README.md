# Aspect-Based Sentiment Analysis for Vietnamese E-commerce Reviews

> Weak Supervision and PhoBERT-based Aspect Sentiment Classification for Vietnamese Tiki.vn Reviews

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PhoBERT](https://img.shields.io/badge/PhoBERT-Vietnamese_NLP-green)
![Snorkel](https://img.shields.io/badge/Snorkel-Weak_Supervision-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# Overview

This project focuses on **Aspect-Based Sentiment Analysis (ABSA)** for Vietnamese e-commerce reviews collected from Tiki.vn. Unlike traditional sentiment analysis that predicts a single sentiment for an entire review, ABSA identifies sentiment toward specific product aspects such as:

* Price
* Quality
* Delivery
* Packaging
* Service
* Authenticity

The project explores both:

* Weak supervision using Snorkel Labeling Functions
* Supervised fine-tuning using PhoBERT

Three approaches were systematically evaluated:

1. Majority Vote / Rule-based labeling
2. Snorkel + PhoBERT
3. Fully supervised PhoBERT with manual annotations

The final PhoBERT model trained on manually annotated reviews achieved:

* **Macro-F1 = 0.584**
* **+16.3% improvement** over weak-supervision-based approaches



---

# Problem Statement

Vietnamese e-commerce reviews are highly noisy and informal:

* Teencode
* Abbreviations
* Emojis
* Missing grammar
* Mixed sentiments in a single sentence

Example:

```text id="ltjz4j"
"ship nhanh, sp ok, bh 12t, đáng đồng tiền"
```

This single review simultaneously expresses:

* Delivery → POS
* Quality → POS
* Service → POS
* Price → POS

Traditional document-level sentiment classification cannot capture these aspect-level signals.



---

# Dataset Collection

Reviews were collected from:

* Electronics
* Books
* Household appliances
* Fashion
* Multiple product categories on Tiki.vn

## Raw Data Statistics

| Stage                 | Reviews |
| --------------------- | ------- |
| Raw collected reviews | 12,306  |
| After deduplication   | 11,368  |
| After filtering       | 7,269   |
| Gold evaluation set   | 500     |
| Training pool         | 6,769   |



---

# Data Preprocessing

A deterministic Vietnamese text preprocessing pipeline was applied:

* Unicode normalization
* Invisible character removal
* Emoji mapping
* Currency normalization
* Slang expansion
* Teencode normalization
* Fixed-expression substitution

Examples:

```text id="oc0w7s"
ko / k / hông → không
sp → sản phẩm
ship → giao hàng
bh → bảo hành
100k → 100000 đồng
```

Approximately **89.3%** of reviews were modified by preprocessing.



---

# Aspect & Label Design

## Product Aspects

* PRICE
* QUALITY
* DELIVERY
* PACKAGING
* SERVICE
* AUTHENTICITY

## Sentiment Labels

| Label | Meaning              |
| ----- | -------------------- |
| POS   | Positive sentiment   |
| NEG   | Negative sentiment   |
| NEU   | Neutral mention      |
| NONE  | Aspect not mentioned |



---

# Gold Annotation

A 500-review gold dataset was manually annotated and fully held out from training.

Annotation guidelines were carefully designed for each aspect:

* DELIVERY refers to shipping process
* SERVICE refers to seller/customer support
* PACKAGING refers to external wrapping/box
* QUALITY refers to product condition itself

This separation was critical for reducing label ambiguity.



---

# Weak Supervision with Snorkel

To reduce manual labeling cost, the project implemented **14 Labeling Functions (LFs)** using Snorkel.

Each LF outputs:

* POS
* NEG
* NEU
* NONE
* ABSTAIN

The LabelModel aggregates noisy LF votes into probabilistic labels.



---

# Labeling Functions

## Initialization Stage

* LF1: Aspect keyword + sentiment window
* LF2: Rating heuristic
* LF3/LF4: Contrast pattern detection

## Domain-Specific Refinement

* Strong polarity lexicons
* Authenticity mismatch detection
* Packaging vs quality damage context
* Shop/service context analysis
* Neutral signal detection

Examples:

```text id="ys4c6x"
"giao nhanh" → DELIVERY POS
"kém chất lượng" → QUALITY NEG
"đóng gói cẩn thận" → PACKAGING POS
"không giống hình" → AUTHENTICITY NEG
```



---

# Snorkel Label Aggregation

Two aggregation strategies were evaluated:

## Majority Vote (MV)

Simple plurality voting from labeling functions.

## Snorkel LabelModel

Learns:

* LF accuracy
* LF coverage
* LF correlations

Outputs soft probability distributions:

```text id="nsv9nl"
[P(POS), P(NEG), P(NEU), P(NONE)]
```

for every aspect.



---

# PhoBERT Architecture

The project uses **PhoBERT-base-v2** with a multi-task classification architecture.

## Pipeline

* Shared PhoBERT encoder
* 6 independent classification heads
* 4-class prediction per aspect

Input text is word-segmented using:

* VnCoreNLP



---

# Training Strategies

## 1. Snorkel-LM

Direct evaluation of Snorkel-generated labels.

## 2. Snorkel + PhoBERT

PhoBERT fine-tuned using Snorkel soft labels with KL-divergence loss.

Training size:

* 5,414 reviews

## 3. PhoBERT (Manual Labels)

PhoBERT trained on:

* 2,990 manually annotated reviews

using weighted cross-entropy loss.



---

# Hyperparameter Optimization

Optuna Bayesian Optimization was used before full training.

## Best Hyperparameters

| Hyperparameter      | Value    |
| ------------------- | -------- |
| Learning Rate       | 7.469e-5 |
| Batch Size          | 8        |
| Warmup Steps        | 350      |
| Dropout             | 0.333    |
| Weight Decay        | 0.05     |
| Max Sequence Length | 256      |
| Epochs              | 15       |



---

# Experimental Results

## Overall Macro-F1 Comparison

| Method                  | Macro-F1  |
| ----------------------- | --------- |
| Majority Vote           | 0.403     |
| Snorkel-LM              | 0.416     |
| Snorkel + PhoBERT       | 0.421     |
| PhoBERT (Manual Labels) | **0.584** |



---

# Key Findings

## Weak Supervision Limitation

Snorkel-based methods struggled because:

* Rule-based labels do not fully match human annotations
* Distributional shift exists between LF-generated labels and real human sentiment interpretation

## NEU Class Collapse

Weak supervision approaches produced:

* NEU F1 = 0.00

for almost all aspects due to insufficient neutral training signals.

## Label Quality > Data Quantity

PhoBERT trained on:

* 2,990 manual labels

outperformed:

* 5,414 Snorkel labels

showing that high-quality labels are more important than large noisy datasets.



---

# Technologies Used

* Python
* PyTorch
* PhoBERT
* Snorkel
* Optuna
* FastAPI
* VnCoreNLP
* Pandas
* Scikit-learn

---

# Project Pipeline

```text id="xj4a48"
Crawl Tiki reviews
        ↓
Data filtering & deduplication
        ↓
Vietnamese preprocessing
        ↓
Gold dataset extraction
        ↓
Weak supervision labeling functions
        ↓
Snorkel LabelModel aggregation
        ↓
PhoBERT fine-tuning
        ↓
Evaluation on gold test set
        ↓
Error analysis
```


# Future Improvements

* Complete annotation of remaining ~3,769 reviews
* Improve NEU class detection
* Explore attention-based aspect extraction
* Multi-label span-based ABSA
* Larger Vietnamese language models
