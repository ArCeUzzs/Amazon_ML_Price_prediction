<!-- -------------------------------------------------- -->
<!-- 🎯 SMART PRODUCT PRICING | ML CHALLENGE 2025 -->
<!-- -------------------------------------------------- -->

<p align="center">
  <img src="https://img.shields.io/badge/Challenge-ML_Challenge_2025-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-LightGBM-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-HuggingFace-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Encoders-BERT_&_CLIP-green?style=for-the-badge"/>
</p>

---

# 🚀 Smart Product Pricing | ML Challenge 2025
*

---

## 🌟 Introduction

In today’s e-commerce landscape, accurately estimating product prices is critical for maintaining competitive advantage.  
The **ML Challenge 2025** tasked participants with predicting product prices using **multimodal data** — combining **textual product descriptions**, **images**, and **structured numerical features**.


I have developed a robust and reproducible **multimodal price prediction pipeline** that leverages the strengths of **language models, vision encoders, and classical machine learning**.  
This system efficiently learns semantic, visual, and quantitative signals to generate reliable price estimates even in the presence of noisy or missing data.
I achieved a **rank of 127 out of over 85000** participating teams

## 🧾 Executive Summary

This project introduces a **multimodal product price prediction system** for **ML Challenge 2025**.

Our solution integrates:
- 📝 **Text embeddings** from a fine-tuned **BERT model**
- 🖼️ **Image embeddings** from a fine-tuned **CLIP ViT-B/32**
- 🔢 **Engineered numerical features**

All these features are **fused and processed** using a **LightGBM regressor** with **5-fold cross-validation**, achieving robust and generalizable results.

📊 **Key Metrics**

| Metric | Score |
|:-------:|:------:|
| **SMAPE** | 30.81 % |
| **MAE (original scale)** | $6.74 |

---

## ⚙️ Methodology

### 🔍 Problem Analysis & Feature Engineering

Dataset: **75,000+ samples** containing text (`catalog_content`), image links, and prices.

**EDA Findings:**
- 🎯 **Target Distribution:** Highly right-skewed → applied `log1p(price)` transformation.  
- 🧠 **Textual Data:** Contained strong brand and quantity indicators.  
- 🖼️ **Image Data:** Added valuable but noisy visual cues (size/packaging).  
- 🔢 **Numerical Features:** Extracted fields such as `total_weight_g`, `pack_qty`, and `pieces` via regex.

---
## 📂 Project Folder & Resources

All project files, artifacts, datasets, notebooks, and your final report can be accessed here:

➡️ [Google Drive – Project Folder](https://drive.google.com/drive/folders/12GJn8kYkXjhpf4YHP2AqRR1LMREOqhOy?usp=sharing)
---

### 🧠 Solution Pipeline

```mermaid
graph TD
    A[Text Data] --> B[BERT Encoder]
    B --> F[Text Embeddings 768D]
    C[Image Data] --> D[CLIP ViT-B32 Encoder]
    D --> G[Image Embeddings 512D]
    E[Numerical Features] --> H[Standard Scaler]
    F --> I[Feature Fusion]
    G --> I
    H --> I
    I --> J[LightGBM Regressor]
    J --> K[Predicted log1p Price]
