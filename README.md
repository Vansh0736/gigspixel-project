# Distributed Systems & Gigapixel AI for Pathology


<img width="1318" height="565" alt="image" src="https://github.com/user-attachments/assets/8f041fb9-51a0-45cf-977b-d27e6104f657" />

# Gigapixel AI: Attention-Based Multiple Instance Learning for Computational Pathology

**Author:** Vansh Jain
**Domain:** Medical AI / Computational Pathology 
**Tech Stack:** PyTorch, TorchVision, NumPy, OpenSlide, Scikit-Learn

##  Project Overview
This repository contains an end-to-end deep learning pipeline for the binary classification (Tumor vs. Normal) of Whole Slide Images (WSIs) in medical pathology. 

Unlike standard computer vision datasets (e.g., ImageNet), pathology WSIs are incredibly massive—often exceeding **50,000 $\times$ 50,000 pixels**. This project successfully diagnoses tumors in these gigapixel images by overcoming extreme hardware constraints using an **Attention-Based Multiple Instance Learning (ABMIL)** architecture.

---

##  The Core Bottleneck: The Gigapixel Memory Wall
**The Problem:** Standard Convolutional Neural Networks (ResNet, VGG) are designed for images ranging from $224 \times 224$ to $1024 \times 1024$ pixels. 
Attempting to pass a raw $50,000 \times 50,000$ WSI through a standard CNN causes an immediate **CUDA Out-Of-Memory (OOM) error** on any modern GPU. Downsampling the image to fit into memory destroys the microscopic cellular level details required to identify malignant tumors.

**The Diagnostic Challenge:** A slide may contain billions of pixels, but a tumor might only occupy a microscopic fraction of that space. If the model predicts "Tumor," it must base that prediction on specific, localized anomalous cellular structures, not background tissue.

---

##  The Solution: Architectural Paradigm Shift to ABMIL
To bypass the memory wall, I discarded the monolithic CNN approach and implemented a **Multiple Instance Learning (MIL)** paradigm. 

Instead of treating the slide as one giant image, the architecture treats the WSI as a "Bag" of smaller patches (instances). 

### The Pipeline Architecture:
1. **Dynamic Tiling (Preprocessing):** * The gigapixel WSI is dynamically sliced into thousands of smaller, manageable $256 \times 256$ patches.
   * Background (empty white space) is filtered out using Otsu's thresholding to save compute cycles.

2. **Independent Feature Extraction:**
   * Each $256 \times 256$ patch is passed independently through a highly optimized CNN backbone (Feature Extractor) to generate a dense latent representation vector. 

3. **Attention-Based Aggregation (The Breakthrough):**
   * *Standard MIL* assumes that if *one* patch is malignant, the whole bag is malignant (Max-Pooling). However, this creates massive false-positive rates due to uncurated dataset noise.
   * *My Architecture* utilizes a custom **Attention Mechanism**. A secondary neural network evaluates the latent vector of every single patch and assigns it an **Attention Score** (weighting its importance).
   * The network learns to naturally "ignore" healthy tissue (low attention) and heavily weight malignant cells (high attention).

4. **Slide-Level Classification:**
   * The attention-weighted patch vectors are aggregated into a single slide-level representation and passed through a final classification head to output the tumor probability.

---

##  Results & Performance Evaluation

The model was evaluated on highly uncurated, real-world pathological datasets containing extensive staining variations and imaging artifacts.

* **Validation AUC:** **0.8537**
* **Performance Note:** The 0.8537 Area Under the Curve demonstrates a highly robust specificity. The attention mechanism successfully isolated tumor signatures without being distracted by artifacts, minimizing false positives despite the massive spatial size of the input data.

---

##  Key Engineering Takeaways
* **Compute Efficiency:** By utilizing ABMIL and aggressive background filtering, inference can be run on gigapixel images using standard consumer GPUs without VRAM exhaustion.
* **Interpretability:** The Attention Scores inherently provide a "heat map" of the WSI. Pathologists can visually inspect which specific $256 \times 256$ patches the neural network deemed "most malignant," providing critical explainability in medical AI.

---
