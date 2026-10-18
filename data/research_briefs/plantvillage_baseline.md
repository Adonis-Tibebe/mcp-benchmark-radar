# PlantVillage Diagnostic Baselines

PlantVillage benchmarks evaluate leaf disease detection across tomato, potato, and bell pepper crops.
MobileNetV2 acts as our low-compute mobile edge baseline but demonstrates acute failure under natural field lighting and heavy outdoor shadow occlusion.
ResNet50 with photometric augmentation delivers 91.5% accuracy, yet suffers high false-negative rates on early-stage bacterial spot lesions smaller than 5 pixels.
Future iterations should evaluate lightweight vision transformers or self-supervised representations to handle real-world field clutter without parameter bloat.
