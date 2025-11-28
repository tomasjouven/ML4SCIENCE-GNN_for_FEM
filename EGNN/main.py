"""Point d'entrée principal du programme."""

import torch
from config import Config
from data_loading import load_data
from model import create_model
from training import train_model, plot_losses


def main():
    """Fonction principale."""
    # Configuration
    config = Config()
    
    # Chargement des données
    train_loader, val_loader, num_train_graphs, num_val_graphs = load_data(config)
    
    # Création du modèle
    model = create_model(config)
    
    # Optimiseur et scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.SCHEDULER_FACTOR, 
        patience=config.SCHEDULER_PATIENCE, 
        verbose=True
    )
    
    # Entraînement
    train_losses, val_losses, best_val_loss = train_model(
        model, train_loader, val_loader, optimizer, scheduler, config,
        num_train_graphs, num_val_graphs
    )
    
    # Visualisation
    plot_losses(train_losses, val_losses, config.TRAINING_CURVE_PATH)


if __name__ == "__main__":
    main()

