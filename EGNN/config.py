"""Configuration et hyperparamètres du modèle."""

import torch

class Config:
    # Données
    TRAIN_DATA_PATH = 'data/train_data.pt'
    VAL_DATA_PATH = 'data/val_data.pt'
    TRAIN_SUBSET_RATIO = 0.1  # Utiliser 20% des données d'entraînement
    BATCH_SIZE = 4
    
    # Architecture
    IN_NODE_FEATURES = 9
    POS_DIM = 3
    HIDDEN_CHANNELS = 16
    OUT_STRESS_DIM = 1
    DEPTH = 1
    POOL_RATIOS = 0.3

    LAMBDA_STRESS = 0.5    
    

    #EGNN Physics
    NUM_NEIGHBORS = 16
    UPDATE_COORS = False    
    UPDATE_FEATS = True
    
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
    TRAINING_CURVE_PATH = f"Curves/training_curve_R{TRAIN_SUBSET_RATIO}_B{BATCH_SIZE}_H{HIDDEN_CHANNELS}_D{DEPTH}_P{POOL_RATIOS}_E{NUM_EPOCHS}.png"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')