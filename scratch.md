# 🧠 Project Notes

## 📅 Date

**19 May 2025 Time 13:39**

## 📌 Overview

Result on 6 csv files Date

## 💡 Logs & Observations

=== Process Started ===
Found 6 CSV files in './dataset/AllData/'
Index(['Src Port', 'Dst Port', 'Protocol', 'Flow Duration', 'Total Fwd Packet',
'Total Bwd packets', 'Total Length of Fwd Packet',
'Total Length of Bwd Packet', 'Fwd Packet Length Max',
'Fwd Packet Length Min', 'Fwd Packet Length Mean',
'Fwd Packet Length Std', 'Bwd Packet Length Max',
'Bwd Packet Length Min', 'Bwd Packet Length Mean',
'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes',
'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
'Idle Min', 'Label'],
dtype='object')

=== Pretraining Phase ===
Pretrain Epoch [1/5], Loss: 1.6896
Pretrain Epoch [2/5], Loss: 1.4663
Pretrain Epoch [3/5], Loss: 1.3436
Pretrain Epoch [4/5], Loss: 1.2072
Pretrain Epoch [5/5], Loss: 1.0678
✅ The file 'pretrained_tabular_model.pth' exists!

=== Fine-tuning Phase ===
Fine-tune Epoch [1/10], Loss: 2.4103, Accuracy: 9.50%
Fine-tune Epoch [2/10], Loss: 2.3821, Accuracy: 11.70%
Fine-tune Epoch [3/10], Loss: 2.3379, Accuracy: 13.20%
Fine-tune Epoch [4/10], Loss: 2.3129, Accuracy: 14.60%
Fine-tune Epoch [5/10], Loss: 2.2705, Accuracy: 14.20%
Fine-tune Epoch [6/10], Loss: 2.2663, Accuracy: 15.80%
Fine-tune Epoch [7/10], Loss: 2.2373, Accuracy: 17.70%
Fine-tune Epoch [8/10], Loss: 2.2208, Accuracy: 18.10%
Fine-tune Epoch [9/10], Loss: 2.2021, Accuracy: 18.90%
Fine-tune Epoch [10/10], Loss: 2.1867, Accuracy: 19.40%
✅ The file 'fine_tuned_tabular_model.pth' exists!

[1] Partitioning data using Dirichlet...

[2] Fleet-level local training...
Train and Test samples: 14656
Train and Test samples: 38189
Train and Test samples: 47670
Train and Test samples: 61399
Train and Test samples: 75077
Train and Test samples: 106175
Train and Test samples: 141643
Train and Test samples: 150492
Train and Test samples: 187927
Train and Test samples: 0

[3] Aggregating fleet models...

[4] Evaluating global model...
Global Accuracy: 0.5000
Client 0 Accuracy: 0.0089
Client 1 Accuracy: 0.5756
Client 2 Accuracy: 0.5567
Client 3 Accuracy: 0.5750
Client 4 Accuracy: 0.5001
Client 5 Accuracy: 0.3536
Client 6 Accuracy: 0.5009
Client 7 Accuracy: 0.5124
Client 8 Accuracy: 0.5562
Client 9 Accuracy: 0.0000

[5] Personalizing each client...
Train and Test samples: 14656
Personalized model trained for Client 0
Train and Test samples: 38189
Personalized model trained for Client 1
Train and Test samples: 47670
Personalized model trained for Client 2
Train and Test samples: 61399
Personalized model trained for Client 3
Train and Test samples: 75077
Personalized model trained for Client 4
Train and Test samples: 106175
Personalized model trained for Client 5
Train and Test samples: 141643
Personalized model trained for Client 6
Train and Test samples: 150492
Personalized model trained for Client 7
Train and Test samples: 187927
Personalized model trained for Client 8
Train and Test samples: 0
Personalized model trained for Client 9

[6] Evaluating personalized models...
Global Accuracy: 0.9900
Personalized Accuracy for Client 0: 0.9900
Global Accuracy: 0.9006
Personalized Accuracy for Client 1: 0.9006
Global Accuracy: 0.8976
Personalized Accuracy for Client 2: 0.8976
Global Accuracy: 0.9049
Personalized Accuracy for Client 3: 0.9049
Global Accuracy: 0.9011
Personalized Accuracy for Client 4: 0.9011
Global Accuracy: 0.9074
Personalized Accuracy for Client 5: 0.9074
Global Accuracy: 0.9144
Personalized Accuracy for Client 6: 0.9144
Global Accuracy: 0.9128
Personalized Accuracy for Client 7: 0.9128
Global Accuracy: 0.9197
Personalized Accuracy for Client 8: 0.9197
Global Accuracy: 0.0000
Personalized Accuracy for Client 9: 0.0000

Process finished with exit code 0

## 📅 Date

****

## 📌 Overview

## 💡 Logs & Observations





