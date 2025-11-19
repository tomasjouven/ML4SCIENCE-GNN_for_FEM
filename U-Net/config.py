"""Configuration et hyperparamètres du modèle."""

import torch

class Config:
    # Données
    TRAIN_DATA_PATH = 'data/train_data.pt'
    VAL_DATA_PATH = 'data/val_data.pt'
    TRAIN_SUBSET_RATIO = 0.2  # Utiliser 20% des données d'entraînement
    BATCH_SIZE = 1
    
    # Architecture
    IN_CHANNELS = 12
    HIDDEN_CHANNELS = 16
    OUT_CHANNELS = 1
    DEPTH = 1
    POOL_RATIOS = 0.3
    
    # Entraînement
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0
    
    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    
    # Sauvegarde
    BEST_MODEL_PATH = 'best_model.pt'
    TRAINING_CURVE_PATH = 'training_curve.png'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')