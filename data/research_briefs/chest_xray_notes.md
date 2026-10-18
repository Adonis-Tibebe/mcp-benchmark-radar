# Medical Chest X-Ray Multi-Label Pathology Notes

Covers multi-label thoracic pathology detection (pneumonia, cardiomegaly, effusion, and atelectasis) from frontal radiographs.
DenseNet-121 initialized with ImageNet weights provides an established clinical baseline achieving 0.864 AUC.
Error analysis reveals that overlapping anatomical structures—specifically rib cage density obscuring retrocardiac airspace—cause missed faint pulmonary consolidations.
Cross-hospital domain shift between imaging devices remains an open challenge, motivating research into multi-scale attention mechanisms and self-supervised chest radiography pretraining.
