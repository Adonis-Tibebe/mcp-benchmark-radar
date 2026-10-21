# Upgrade Plan: Amharic OCR Performance Improvement

## Current Baseline Comparison
| Model | Accuracy | F1 Score | Params (M) |
|---|---|---|---|
| CRNN-CTC-v1 | 0.714 | 0.698 | 8.2 |
| BiLSTM-Attention | 0.768 | 0.752 | 14.1 |

## Analysis of Failure Modes
The current models struggle with:
1. **Glyph Confusion:** Difficulty distinguishing between visually similar Ge'ez characters (e.g., 'ሀ', 'ሁ', 'ሂ').
2. **Resolution Sensitivity:** Performance drops significantly below 200 DPI.
3. **Noise Robustness:** Susceptibility to bleed-through and document degradation.

## Proposed Upgrade: Vision Transformer (ViT) + Encoder-Decoder
To overcome these limitations, we propose moving away from pure RNN-based architectures toward a Transformer-based approach (e.g., TrOCR or similar Vision-Language models).

### Key Improvements:
* **Global Context:** Unlike RNNs, Transformers capture global dependencies, which helps in distinguishing subtle character variations by looking at the broader context of the word/sentence.
* **Robustness:** Pre-training on large-scale synthetic datasets (using diverse fonts and noise augmentation) will improve performance on low-resolution and degraded documents.
* **Attention Mechanisms:** Moving from simple Bahdanau attention to multi-head self-attention allows the model to better focus on fine-grained features of the Ethiopic script.

## Action Plan
1. **Data Augmentation:** Implement synthetic data generation pipeline including blur, noise, and bleed-through simulations.
2. **Architecture Shift:** Replace the BiLSTM backbone with a ViT encoder and a Transformer decoder.
3. **Fine-tuning:** Fine-tune on high-quality, high-resolution Amharic datasets to establish a new baseline.
