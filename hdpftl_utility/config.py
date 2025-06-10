# Directory Structure (suggested)
# hdpftl/
# ├── __init__.py
# ├── config.py.py
# ├── hdpftl_data/
# │   ├── __init__.py
# │   ├── downloader.py
# │   └── preprocess.py
# ├── hdpftl_models/
# │   ├── __init__.py
# │   └── TabularNet.py
# ├── hdpftl_training/
# │   ├── __init__.py
# │   ├── local_train.py
# │   └── federated.py
# ├── evaluate.py
# ├── hdpftl_main.py
# └── utils.py
import os
from string import Template

# Main Code Flow Based on the Diagrams
# 🔁 Level 1: Device-Level Training (Personalized Learning)
# Collect device-specific hdpftl_data
#
# Train personalized local hdpftl_models (train_device_model)
#
# Optionally evaluate and fine-tune with historical normal hdpftl_data to detect anomalies
#
# 🔁 Level 2: Fleet-Level Aggregation (Edge Aggregation)
# Send model weights to fleet edge aggregator
#
# Aggregate weights using aggregate_models(hdpftl_models) → output: fleet model
#
# Broadcast fleet model back to devices for local adaptation
#
# 🔁 Level 3: Global Aggregation (Cross-Silo Transfer)
# Multiple fleet hdpftl_models (from different silos) are aggregated at the cloud/global level
#
# Use aggregate_models() again → output: global model
#
# Optionally fine-tune global model with pre-hdpftl_training learning on target silo's hdpftl_data


# 1. Data Preprocessing per Silo
# Load raw hdpftl_data (e.g., CICIDS2017) for each IoT silo.
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
# Freeze lower layers (for pre-hdpftl_training learning)
#
# 3. Personalized Training per Client (IoT Silo)
# Each silo:
#
# Fine-tunes its local model on local (non-IID) hdpftl_data
#
# Uses pre-hdpftl_training learning to adapt the shared model
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
# Edge nodes send their aggregated hdpftl_models to the cloud
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
OUTPUT_DATASET_PATH_2023 = "./hdpftl_dataset/CIC_IoT_IDAD_Dataset_2023/"
OUTPUT_DATASET_PATH_2024 = "./hdpftl_dataset/CIC_IoT_IDAD_Dataset_2024/"
OUTPUT_DATASET_ALL_DATA = "./hdpftl_training/hdpftl_dataset/AllData/"

BATCH_SIZE = 5
NUM_CLIENTS = 7
NUM_DEVICES_PER_CLIENT = 5
CLIENTS_PER_AGGREGATOR = 5
NUM_ROUNDS = 5
input_dim = 79  # Your feature size
NUM_EPOCHS_PRE_TRAIN = 10  # or 50 or 100
NUM_EPOCHS_FINE_TUNE = 20  # or 50 or 100

GLOBAL_MODEL_PATH = "./hdpftl_trained_models/global_model.pth"
FINETUNE_MODEL_PATH = "./hdpftl_trained_models/fine_tuned_tabular_model.pth"
PRE_MODEL_PATH = "./hdpftl_trained_models/pretrained_tabular_model.pth"
EPOCH_DIR_FINE = "./hdpftl_trained_models/epochs"
EPOCH_FILE_FINE = os.path.join(EPOCH_DIR_FINE, "fine_tune_epoch_losses.npy")
EPOCH_DIR_PRE = "./hdpftl_trained_models/epochs"
EPOCH_FILE_PRE = os.path.join(EPOCH_DIR_PRE, "pre_epoch_losses.npy")

PLOT_PATH = "./hdpftl_plot_outputs/"
PERSONALISED_MODEL_PATH_TEMPLATE = Template("./hdpftl_trained_models/personalized_model_client_${n}.pth")
