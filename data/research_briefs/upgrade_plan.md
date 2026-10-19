# ML Benchmark Upgrade Plan & References

This research brief outlines upgrade strategies for datasets currently flagged as `Needs Upgrade` in our benchmark database (`data/benchmarks.db`), complete with GitHub repository references, web search contexts, and notes on missing or unretrieved contexts.

---

## 1. PlantVillage (Image Classification)
* **Current Baseline:** MobileNetV2-Baseline (Accuracy: 0.842, F1: 0.835)
* **Active Model:** ResNet50-Augmented (Accuracy: 0.915, F1: 0.910)
* **Upgrade Strategy:** Fine-tune EfficientNetB0/B4 with advanced data augmentation (RandAugment, Mixup) and Gradio deployment for interactive plant disease classification.
* **GitHub References:**
  * [`Sowaiba-01/AgroVision-Net`](https://github.com/Sowaiba-01/AgroVision-Net): Implements EfficientNetB0 fine-tuning on the PlantVillage dataset across 38 categories with Gradio web interface and confidence thresholding.
  * [`spMohanty/PlantVillage-Dataset`](https://github.com/spMohanty/PlantVillage-Dataset): Original dataset source repository.

---

## 2. CreditCard-Fraud (Tabular Classification)
* **Current Baseline:** XGBoost-Default (Accuracy: 0.981, F1: 0.740 — *Note: high accuracy is misleading due to extreme class imbalance; F1/AUPRC is critical*).
* **Upgrade Strategy:** Implement SMOTE / ADASYN oversampling, cost-sensitive learning, and anomaly detection models (Isolation Forest, Autoencoders) to improve minority class F1 score.
* **GitHub References:**
  * [`Kaiiser7/Credit-Card-Fraud-Detection-using-Machine-Learning-with-Python`](https://github.com/Kaiiser7/Credit-Card-Fraud-Detection-using-Machine-Learning-with-Python): Standard tabular pipeline referencing the ULB Machine Learning Group benchmark and the [Machine Learning for Credit Card Fraud Detection Practical Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/).
  * [`Manukondaindhu/Manukondaindhu-Growfinix-Task-2`](https://github.com/Manukondaindhu/Manukondaindhu-Growfinix-Task-2): Demonstrates the integration of Scikit-Learn, Isolation Forest, and SMOTE for imbalanced tabular fraud detection.

---

## 3. CIFAR-100 (Image Classification)
* **Current Baseline:** SimpleCNN-4Layer (Accuracy: 0.625, F1: 0.618)
* **Active Model:** ViT-Tiny-Patch16 (Accuracy: 0.789, F1: 0.784)
* **Upgrade Strategy:** Transition to ConvNeXt or Swin Transformer backbones with cosine learning rate schedules and strong regularization (CutMix, Label Smoothing).
* **Missing / Unretrieved Context Notes:**
  * *Web & GitHub Search Status:* Direct repository queries for `"CIFAR-100 state of the art models github repos"` returned no direct matches via the automated search tool. SOTA benchmarking for CIFAR-100 relies on standard academic repositories (e.g., Hugging Face `timm` models, PyTorch Image Models).

---

## 4. Amharic-OCR (Optical Character Recognition)
* **Current Baseline:** CRNN-CTC-v1 (Accuracy: 0.714, F1: 0.698)
* **Active Model:** BiLSTM-Attention (Accuracy: 0.768, F1: 0.752)
* **Upgrade Strategy:** Upgrade to Vision Transformer (TrOCR) or CRNN + Transformer Decoder architectures tailored for Ethiopic script character recognition.
* **Missing / Unretrieved Context Notes:**
  * *Web & GitHub Search Status:* Specific open-source repositories for Amharic-OCR deep learning baselines were not returned by current GitHub REST searches. Most implementations remain localized or private academic research codebases.

---
*Synthesized and verified by AI Research & Benchmark Radar.*
