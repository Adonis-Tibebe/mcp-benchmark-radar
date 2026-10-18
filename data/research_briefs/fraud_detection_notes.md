# Credit Card Fraud Detection Diagnostics

Assesses transaction anomaly classification under extreme class imbalance with approximately 0.17% positive fraudulent cases.
XGBoost default parameters yield an apparently strong accuracy of 98.1%, but mask a sub-optimal F1 score of 0.740 caused by high false-negative rates on novel zero-day attack vectors.
Feature correlation shows high vulnerability to burst transaction velocities and foreign currency conversions.
Recommended upgrades include focal loss tuning, graph neural network topology embeddings for cardholder-merchant graphs, or unsupervised autoencoder reconstruction filtering.
