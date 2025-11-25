"""Boucle d'entraînement et validation."""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


def train_epoch(model, train_loader, optimizer, device, gradient_clip, num_train_graphs):
    """
    Effectue une epoch d'entraînement.
    
    Returns:
        train_loss: Loss moyenne sur l'epoch
    """
    model.train()
    train_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(out, batch.y)
        loss.backward()
        
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs
    
    train_loss /= num_train_graphs
    return train_loss


def validate(model, val_loader, device, num_val_graphs):
    """
    Effectue la validation.
    
    Returns:
        val_loss: Loss moyenne sur la validation
    """
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(out, batch.y)
            val_loss += loss.item() * batch.num_graphs
    
    val_loss /= num_val_graphs
    return val_loss


def train_model(model, train_loader, val_loader, optimizer, scheduler, config, 
                num_train_graphs, num_val_graphs):
    """
    Boucle d'entraînement complète.
    
    Returns:
        train_losses, val_losses, best_val_loss
    """
    print("="*70)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("="*70 + "\n")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        # Entraînement
        train_loss = train_epoch(
            model, train_loader, optimizer, config.DEVICE, 
            config.GRADIENT_CLIP, num_train_graphs
        )
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate(model, val_loader, config.DEVICE, num_val_graphs)
        val_losses.append(val_loss)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Sauvegarde du meilleur modèle
        epoch_time = time.time() - start_time
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, config.BEST_MODEL_PATH)
            print(f"Epoch {epoch}/{config.NUM_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {epoch_time:.3f}s | LR: {optimizer.param_groups[0]['lr']:.2e} ✓ Best!")
        else:
            print(f"Epoch {epoch}/{config.NUM_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {epoch_time:.3f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    print(f"\n🎯 Best validation loss: {best_val_loss:.6f}")
    print("="*70 + "\n")
    
    return train_losses, val_losses, best_val_loss


def plot_losses(train_losses, val_losses, save_path):
    """
    Trace et sauvegarde les courbes de loss.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Training & Validation Loss", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Courbe de loss sauvegardée: {save_path}")
    plt.show()