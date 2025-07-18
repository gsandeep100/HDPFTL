# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        downloader.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-23
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import os

import requests


def download_dataset(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Fake browser headers
    # Send a GET request to the URL
    response = requests.get(input_dir)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the content to a local file
        with open('CIC_IoT_Dataset_2023.html', 'wb') as file:
            file.write(response.content)
        safe_log('File downloaded successfully.')
    else:
        safe_log(f'Failed to download file. Status code: {response.status_code}', level="error")

# def download_dataset():
#     if not os.path.exists(DATASET_PATH_2023):
#         api = KaggleApi()
#         api.authenticate()
#         path = kagglehub.dataset_download("bertvankeulen/cicids-2017")
#         #path1 = kagglehub.dataset_download("ishasingh03/friday-workinghours-afternoon-ddos")
#         shutil.copytree(path, DATASET_PATH_2023)
#         #shutil.copytree(path1, DATASET_PATH)
#         safe_log("Dataset downloaded at:", DATASET_PATH_2023)
#     else:
#         safe_log("Dataset already exists at:", DATASET_PATH_2023)
