"""Configuration et hyperparamètres du modèle."""

import torch

class Config:
    # Données
    TRAIN_DATA_PATH = 'data/train_data.pt'
    VAL_DATA_PATH = 'data/val_data.pt'
    TRAIN_SUBSET_RATIO = 0.01  # Utiliser 20% des données d'entraînement
    BATCH_SIZE = 1
    
    # Architecture
    IN_CHANNELS = 12
    HIDDEN_CHANNELS = 32
    OUT_CHANNELS = 1
    DEPTH = 3
    POOL_RATIOS = 0.5
    
    # Entraînement
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0 #regu
    GRADIENT_CLIP = 1.0
    
    # Sauvegarde
    BEST_MODEL_PATH = 'best_model.pt'
    TRAINING_CURVE_PATH = 'training_curve.png'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')