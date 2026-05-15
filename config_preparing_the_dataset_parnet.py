# config_preparing_the_dataset_parnet.py
""" Конфигурация для создания парнет-датасета из prepared_dataset. """
import torch

DATASET_DIR = "./prepared_dataset"
OUTPUT_DIR = "./prepared_dataset_parnet"
ENCODER_CHECKPOINT = "./models/encoder_epoch200.pth"
NUM_WORKERS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"