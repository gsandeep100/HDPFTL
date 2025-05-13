# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        evaluate.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-22
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import torch


#✅ Evaluate on Global Test Set
# def evaluate_global_model(model, X_test, y_test, batch_size=32, device='cuda'):
#     model.eval()
#     dataloader = DataLoader(TensorDataset(X_test.to(device), y_test.to(device)), batch_size=batch_size)
#     correct, total = 0, 0
#     with torch.no_grad():
#         for x, y in dataloader:
#             output = model(x)
#             _, pred = torch.max(output, 1)
#             total += y.size(0)
#             correct += (pred == y).sum().item()
#     accuracy = correct / total
#     print(f"Global Model Accuracy: {accuracy:.4f}")
#     return accuracy
#
#
# #Evaluate Per-Client Accuracy (Optional)
# def evaluate_per_client(model, X_train, y_train, client_partitions, batch_size=32, device='cuda'):
#     model.eval()
#     accuracies = {}
#     with torch.no_grad():
#         for client_id, indices in client_partitions.items():
#             x = X_train[indices].to(device)
#             y = y_train[indices].to(device)
#             dataloader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
#             correct, total = 0, 0
#             for xb, yb in dataloader:
#                 output = model(xb)
#                 _, pred = torch.max(output, 1)
#                 total += yb.size(0)
#                 correct += (pred == yb).sum().item()
#             accuracies[client_id] = correct / total
#             print(f"Client {client_id} Accuracy: {accuracies[client_id]:.4f}")
#     return accuracies
#

def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, pred = torch.max(output, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / total
