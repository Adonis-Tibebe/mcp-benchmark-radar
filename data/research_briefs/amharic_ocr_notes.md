# Amharic OCR Benchmark Notes

This benchmark tests sequence recognition of the Ethiopic / Ge'ez syllabary across scanned documents and historical texts.
The CRNN-CTC baseline struggles heavily with character segmentation confusion among visually near-identical glyphs (such as consonant forms with subtle tick modifications like 'ሀ', 'ሁ', and 'ሂ').
BiLSTM with Bahdanau attention raises recognition accuracy to 76.8%, yet deteriorates sharply when document resolution drops below 200 DPI or when bleed-through noise is present.
Upgrading to a 2D vision transformer backbone with modern CTC/Cross-Entropy decoding or a lightweight Vision-Language Model is urgently required.
