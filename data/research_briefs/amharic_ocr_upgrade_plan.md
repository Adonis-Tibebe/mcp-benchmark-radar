# Amharic-OCR Upgrade Plan & Research Brief

## 1. Baseline Overview
- **Dataset:** Amharic-OCR
- **Current Baselines:**
  - `CRNN-CTC-v1`: Accuracy 0.714, F1 0.698, 8.2M params (`Needs Upgrade`)
  - `BiLSTM-Attention`: Accuracy 0.768, F1 0.752, 14.1M params (`Baseline`)
- **Key Failure Modes:** 
  - Character segmentation confusion among visually near-identical Ethiopic/Ge'ez glyphs (e.g. consonant-vowel orders like 'ሀ', 'ሁ', 'ሂ').
  - Sharp degradation when document resolution drops below 200 DPI or when historical bleed-through noise is present.

## 2. External & SOTA Insights
- **Repositories & Tools:** MMOCR integration (`dikubab/Amharic_OCR`) and deep learning pipelines (`MenelikBerhan/amharic_ocr_draft`).
- **Architectural Directions:** Moving beyond 1D sequence models (CRNN/BiLSTM) toward 2D vision transformer (ViT) backbones (e.g. TrOCR, Donut) combined with robust CTC / autoregressive decoding tailored for complex scripts.

## 3. Actionable Upgrade Recommendations
1. **Backbone Upgrade:** Replace the 1D CNN + BiLSTM feature extractor with a Patch-based 2D Vision Transformer (ViT) to better capture complex spatial and topological nuances of Ge'ez character components.
2. **Data Augmentation:** Introduce robust synthetic data augmentations simulating low DPI (downsampling), blur, paper bleed-through, and skewness to tackle historical document noise.
3. **Loss & Decoding Refinement:** Implement a hybrid CTC and Attention-guided beam search decoder to penalize glyph confusions among closely related syllabic forms.
