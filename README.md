# Distributed Systems & Gigapixel AI for Pathology


<img width="1318" height="565" alt="image" src="https://github.com/user-attachments/assets/8f041fb9-51a0-45cf-977b-d27e6104f657" />

## Overview
In computational pathology, analyzing uncurated Whole Slide Images (WSIs) presents a massive challenge due to the "Memory Wall," as a single image often exceeds 50,000 x 50,000 pixels. This project implements a distributed data pipeline and an advanced transformer architecture to perform highly accurate binary classification (Normal vs. Tumor) on gigapixel-scale images without requiring exhaustive pixel-level annotations.

## Architecture & Methodology

### 1. Distributed Data Engineering
To bypass severe hardware timeouts and VRAM limitations, the data extraction pipeline was parallelized across 7 independent GPU nodes. The pipeline utilizes CTransPath, a CNN-Transformer hybrid foundation model, to extract 768-dimensional latent vectors from raw pixel patches. This strategy successfully compressed 500GB of raw WSI data into a unified, manageable 27GB latent space.

### 2. Attention-Based Multiple Instance Learning (ABMIL)
Standard Vision Transformers (ViTs) scale quadratically, which leads to catastrophic Out-Of-Memory (OOM) errors when processing WSIs containing 10,000+ patches. To solve this $O(N^2)$ complexity constraint, this project implements Attention-Based Multiple Instance Learning (ABMIL). By treating the entire WSI as a "bag" of instances and applying Gated Attention, the computational complexity is reduced to Linear $O(N)$.

## Clinical Results
The final diagnostic pipeline achieved highly competitive metrics, successfully isolating the tumor signal from massive spatial noise:
* **Validation AUC:** 0.8537
* **Confusion Matrix Performance:** 29 True Negatives and 17 True Positives, with minimal misclassifications (5 False Negatives, 3 False Positives).
* **Deployment Efficiency:** The entire diagnostic intelligence was exported into an ultra-lightweight 3.26 MB portable weight file, highly optimized for low-resource deployment.
