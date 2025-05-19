# Directory Structure (suggested)
# hdpftl/
# ├── __init__.py
# ├── config.py.py
# ├── data/
# │   ├── __init__.py
# │   ├── downloader.py
# │   └── preprocess.py
# ├── models/
# │   ├── __init__.py
# │   └── TabularNet.py
# ├── training/
# │   ├── __init__.py
# │   ├── local_train.py
# │   └── federated.py
# ├── evaluate.py
# ├── main.py
# └── utils.py

# Main Code Flow Based on the Diagrams
# 🔁 Level 1: Device-Level Training (Personalized Learning)
# Collect device-specific data
#
# Train personalized local models (train_device_model)
#
# Optionally evaluate and fine-tune with historical normal data to detect anomalies
#
# 🔁 Level 2: Fleet-Level Aggregation (Edge Aggregation)
# Send model weights to fleet edge aggregator
#
# Aggregate weights using aggregate_models(models) → output: fleet model
#
# Broadcast fleet model back to devices for local adaptation
#
# 🔁 Level 3: Global Aggregation (Cross-Silo Transfer)
# Multiple fleet models (from different silos) are aggregated at the cloud/global level
#
# Use aggregate_models() again → output: global model
#
# Optionally fine-tune global model with transfer learning on target silo's data


# 1. Data Preprocessing per Silo
# Load raw data (e.g., CICIDS2017) for each IoT silo.
#
# Apply:
#
# Label encoding / one-hot encoding
#
# Feature scaling (MinMaxScaler / StandardScaler)
#
# Handle imbalance (SMOTE / ADASYN)
#
# Train/test split
#
# Non-IID partitioning (e.g., Dirichlet distribution)
#
# 2. Base Model Setup
# Load a pretrained model (e.g., ResNet-18, or others)
#
# Modify final layers for intrusion detection classes
#
# Freeze lower layers (for transfer learning)
#
# 3. Personalized Training per Client (IoT Silo)
# Each silo:
#
# Fine-tunes its local model on local (non-IID) data
#
# Uses transfer learning to adapt the shared model
#
# Apply regularization to maintain personalization
#
# 4. Local Updates
# Train using FedAvg or FedProx variants
#
# Save local weights
#
# 5. Hierarchical Aggregation
# Edge Level (First Aggregation):
#
# Clients under one regional edge node send updates
#
# Aggregation via weighted average / FedAvg / attention-based fusion
#
# Global Level (Second Aggregation):
#
# Edge nodes send their aggregated models to the cloud
#
# Global model update using hierarchical FedAvg
#
# 6. Model Distribution
# Updated global model pushed down:
#
# To edges
#
# Then to silos (personalized re-init if needed)
#
# 7. Evaluation
# Test each personalized client model on local test set
#
# Log:
#
# Accuracy, Precision, Recall, F1, AUC
#
# Zero-day attack detection performance
#
# 8. Repeat for Several Rounds
# Federated learning is iterative (e.g., 50–100 rounds)
#
# Optional: dynamic client participation, dropout, model staleness
#
# 9. Final Deployment
# Use the final personalized model per silo
#
# Deployed on-device or on-edge for real-time zero-day detection
INPUT_DATASET_PATH_2023 = 'https://www.unb.ca/cic/datasets/iotdataset-2023.html'
INPUT_DATASET_PATH_2024 = 'https://www.unb.ca/cic/datasets/iotdataset-2024.html'
OUTPUT_DATASET_PATH_2023 = "./dataset/CIC_IoT_IDAD_Dataset_2023/"
OUTPUT_DATASET_PATH_2024 = "./dataset/CIC_IoT_IDAD_Dataset_2024/"
OUTPUT_DATASET_ALL_DATA = "./dataset/AllData/"


BATCH_SIZE = 32
NUM_CLIENTS = 10
NUM_DEVICE = 2
CLIENTS_PER_AGGREGATOR = 5
NUM_ROUNDS = 5
input_dim = 79  # Your feature size
pretrain_classes = 5  # Suppose you pretrained with 5 classes
target_classes = 10  # Suppose you pretrained with 5 classes
