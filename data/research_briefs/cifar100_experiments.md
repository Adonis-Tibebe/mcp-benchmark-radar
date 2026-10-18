# CIFAR-100 Fine-Grained Classification

Evaluates fine-grained multi-class recognition across 100 classes grouped into 20 superclasses.
The 4-layer SimpleCNN baseline underfits severely due to restricted receptive field depth and lack of skip connections.
While ViT-Tiny-Patch16 improves top-1 accuracy to 78.9%, confusion matrices reveal severe misclassifications between semantically adjacent sub-classes, particularly aquatic mammals and small carnivores.
Investigating modern hierarchical architectures such as Swin Transformers or ConvNeXt with aggressive mixup augmentation is recommended to pass the 80% mark.
