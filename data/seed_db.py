"""Seed script for AI Research & Benchmark Radar.

Populates `data/benchmarks.db` with deterministic baseline records
and generates initial markdown research briefs inside `data/research_briefs/`.
"""

import sqlite3
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "benchmarks.db"
BRIEFS_DIR = BASE_DIR / "research_briefs"

# Deterministic Benchmark Records
BENCHMARK_RECORDS = [
    (
        1,
        "PlantVillage",
        "MobileNetV2-Baseline",
        "Image Classification",
        0.842,
        0.835,
        3.4,
        "plantvillage_baseline.md",
        "Baseline",
    ),
    (
        2,
        "PlantVillage",
        "ResNet50-Augmented",
        "Image Classification",
        0.915,
        0.910,
        25.6,
        "plantvillage_baseline.md",
        "Active",
    ),
    (
        3,
        "Amharic-OCR",
        "CRNN-CTC-v1",
        "Optical Character Recognition",
        0.714,
        0.698,
        8.2,
        "amharic_ocr_notes.md",
        "Needs Upgrade",
    ),
    (
        4,
        "Amharic-OCR",
        "BiLSTM-Attention",
        "Optical Character Recognition",
        0.768,
        0.752,
        14.1,
        "amharic_ocr_notes.md",
        "Baseline",
    ),
    (
        5,
        "CIFAR-100",
        "SimpleCNN-4Layer",
        "Image Classification",
        0.625,
        0.618,
        1.8,
        "cifar100_experiments.md",
        "Needs Upgrade",
    ),
    (
        6,
        "CIFAR-100",
        "ViT-Tiny-Patch16",
        "Image Classification",
        0.789,
        0.784,
        5.7,
        "cifar100_experiments.md",
        "Active",
    ),
    (
        7,
        "CreditCard-Fraud",
        "XGBoost-Default",
        "Tabular Classification",
        0.981,
        0.740,
        0.5,
        "fraud_detection_notes.md",
        "Needs Upgrade",
    ),
    (
        8,
        "Medical-ChestXRay",
        "DenseNet121-Pretrained",
        "Medical Imaging",
        0.864,
        0.851,
        8.0,
        "chest_xray_notes.md",
        "Baseline",
    ),
]

# Initial Research Briefs explaining failure modes and domain context
RESEARCH_BRIEFS = {
    "plantvillage_baseline.md": (
        "# PlantVillage Diagnostic Baselines\n\n"
        "PlantVillage benchmarks evaluate leaf disease detection across tomato, potato, and bell pepper crops.\n"
        "MobileNetV2 acts as our low-compute mobile edge baseline but demonstrates acute failure under natural field lighting and heavy outdoor shadow occlusion.\n"
        "ResNet50 with photometric augmentation delivers 91.5% accuracy, yet suffers high false-negative rates on early-stage bacterial spot lesions smaller than 5 pixels.\n"
        "Future iterations should evaluate lightweight vision transformers or self-supervised representations to handle real-world field clutter without parameter bloat.\n"
    ),
    "amharic_ocr_notes.md": (
        "# Amharic OCR Benchmark Notes\n\n"
        "This benchmark tests sequence recognition of the Ethiopic / Ge'ez syllabary across scanned documents and historical texts.\n"
        "The CRNN-CTC baseline struggles heavily with character segmentation confusion among visually near-identical glyphs (such as consonant forms with subtle tick modifications like 'ሀ', 'ሁ', and 'ሂ').\n"
        "BiLSTM with Bahdanau attention raises recognition accuracy to 76.8%, yet deteriorates sharply when document resolution drops below 200 DPI or when bleed-through noise is present.\n"
        "Upgrading to a 2D vision transformer backbone with modern CTC/Cross-Entropy decoding or a lightweight Vision-Language Model is urgently required.\n"
    ),
    "cifar100_experiments.md": (
        "# CIFAR-100 Fine-Grained Classification\n\n"
        "Evaluates fine-grained multi-class recognition across 100 classes grouped into 20 superclasses.\n"
        "The 4-layer SimpleCNN baseline underfits severely due to restricted receptive field depth and lack of skip connections.\n"
        "While ViT-Tiny-Patch16 improves top-1 accuracy to 78.9%, confusion matrices reveal severe misclassifications between semantically adjacent sub-classes, particularly aquatic mammals and small carnivores.\n"
        "Investigating modern hierarchical architectures such as Swin Transformers or ConvNeXt with aggressive mixup augmentation is recommended to pass the 80% mark.\n"
    ),
    "fraud_detection_notes.md": (
        "# Credit Card Fraud Detection Diagnostics\n\n"
        "Assesses transaction anomaly classification under extreme class imbalance with approximately 0.17% positive fraudulent cases.\n"
        "XGBoost default parameters yield an apparently strong accuracy of 98.1%, but mask a sub-optimal F1 score of 0.740 caused by high false-negative rates on novel zero-day attack vectors.\n"
        "Feature correlation shows high vulnerability to burst transaction velocities and foreign currency conversions.\n"
        "Recommended upgrades include focal loss tuning, graph neural network topology embeddings for cardholder-merchant graphs, or unsupervised autoencoder reconstruction filtering.\n"
    ),
    "chest_xray_notes.md": (
        "# Medical Chest X-Ray Multi-Label Pathology Notes\n\n"
        "Covers multi-label thoracic pathology detection (pneumonia, cardiomegaly, effusion, and atelectasis) from frontal radiographs.\n"
        "DenseNet-121 initialized with ImageNet weights provides an established clinical baseline achieving 0.864 AUC.\n"
        "Error analysis reveals that overlapping anatomical structures—specifically rib cage density obscuring retrocardiac airspace—cause missed faint pulmonary consolidations.\n"
        "Cross-hospital domain shift between imaging devices remains an open challenge, motivating research into multi-scale attention mechanisms and self-supervised chest radiography pretraining.\n"
    ),
}


def seed_database() -> None:
    """Create benchmarks table and insert deterministic benchmark rows."""
    print(f"Connecting to database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS model_benchmarks;")
    cursor.execute(
        """
        CREATE TABLE model_benchmarks (
            id INTEGER PRIMARY KEY,
            dataset TEXT NOT NULL,
            model_name TEXT NOT NULL,
            task_type TEXT NOT NULL,
            accuracy REAL NOT NULL,
            f1_score REAL NOT NULL,
            params_millions REAL NOT NULL,
            notes_file TEXT NOT NULL,
            status TEXT NOT NULL
        );
        """
    )

    cursor.executemany(
        """
        INSERT INTO model_benchmarks (
            id, dataset, model_name, task_type, accuracy, f1_score, params_millions, notes_file, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        BENCHMARK_RECORDS,
    )

    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM model_benchmarks;")
    count = cursor.fetchone()[0]
    conn.close()
    print(f"Successfully seeded {count} benchmark records into {DB_PATH.name}.")


def seed_research_briefs() -> None:
    """Write initial markdown research briefs to data/research_briefs/."""
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in RESEARCH_BRIEFS.items():
        file_path = BRIEFS_DIR / filename
        file_path.write_text(content, encoding="utf-8")
        print(f"Created research brief: {file_path.name}")
    print(f"Successfully created {len(RESEARCH_BRIEFS)} research briefs in {BRIEFS_DIR}.")


def main() -> None:
    """Run seeding routine."""
    print("--- Seeding AI Research & Benchmark Radar Data ---")
    seed_database()
    seed_research_briefs()
    print("--- Seeding Complete ---")


if __name__ == "__main__":
    main()
