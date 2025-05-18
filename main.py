# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        main.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-21
   Python3 Version:   3.12.8
-------------------------------------------------
"""

from data.preprocess import preprocess_data
from result.final_model import save, load
from training.hdpftl_pipeline import hdpftl_pipeline
from transfer.pretrainclass import pretrain_class
from transfer.targetclass import target_class
from utility.config import OUTPUT_DATASET_PATH_2024


if __name__ == "__main__":
    print("\n=== Process Started ===")
    # download_dataset(INPUT_DATASET_PATH_2024, OUTPUT_DATASET_PATH_2024)
    X_train, X_test, y_train, y_test = preprocess_data(OUTPUT_DATASET_PATH_2024)
    # Step 2: Pretrain global model
    pre_model = pretrain_class()

    # Step 3: Instantiate target model and train on device
    model = target_class()
    global_model, personalized_models = hdpftl_pipeline(X_train, y_train, X_test, y_test, model)
    #save(global_model, personalized_models)
    #load(global_model)

    ########################################
    # model = train_device_model(model, X_train, y_train, setup_device())
    # # Step 4: Aggregate models (e.g., fine-tune with pretrained model)
    # aggregate_models(model, pretrain_class, setup_device())
    # # Step 5: Evaluate
    # evaluate_model(model, X_test, y_test)
    # # Step 6: Optional fleet training
    # model = train_fleet(model)
    ############################################
