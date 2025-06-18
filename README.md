# Directory Structure (suggested)

# hdpftl/

# ├── __init__.py

# ├── config.py.py

# ├── data/

# │ ├── __init__.py

# │ ├── downloader.py

# │ └── preprocess.py

# ├── models/

# │ ├── __init__.py

# │ └── TabularNet.py

# ├── training/

# │ ├── __init__.py

# │ ├── local_train.py

# │ └── federated.py

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

#dirichlet_partition
dirichlet_partition typically refers to a data partitioning strategy that uses the Dirichlet distribution to divide
datasets — especially in federated learning — in order to simulate non-IID (non-identically independently distributed)
data across clients.
📘 What is the Dirichlet Distribution?
The Dirichlet distribution is a multivariate generalization of the Beta distribution. It's often used to generate random
proportions that sum to 1. It’s parameterized by a vector α = [α₁, α₂, ..., α_k].

For federated learning, it controls how skewed the label distribution is across clients.

🎯 Purpose of dirichlet_partition in Federated Learning
It’s used to simulate real-world heterogeneity, where clients (e.g., edge devices or users) have different distributions
of data labels.

Example:
If you have 3 classes (cat, dog, car) and 5 clients:

A Dirichlet α = 1.0 will create roughly balanced distributions.

A Dirichlet α = 0.1 will result in some clients having almost entirely one class — highly non-IID.

# TODO

1. Check for Data leaks on every layer(processing,splitting,training,evaluation)
2. use only finetune data in dirichet splitting instead of full data
3. use Bayseian instead of fedavg