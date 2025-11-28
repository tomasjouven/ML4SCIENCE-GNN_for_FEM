"""Boucle d'entraînement et validation."""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_batch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

        h_dense, mask = to_dense_batch(batch.x, batch.batch)
        pos_dense, _ = to_dense_batch(batch.pos, batch.batch)

        target_stress_dense, _ = to_dense_batch(batch.y, batch.batch)

        pred_stress = model(h_dense, pos_dense, mask=mask)

        loss = F.huber_loss(pred_stress, target_stress_dense, delta=0.5)
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
        
        all_graph_r2s = [] # Pour les percentiles (par géométrie)
        all_predictions_global = [] # Pour RMSE/MAE (global)
        all_targets_global = [] # Pour RMSE/MAE (global)
        for batch in val_loader:
            batch = batch.to(device)
            h_dense, mask = to_dense_batch(batch.x, batch.batch)
            pos_dense, _ = to_dense_batch(batch.pos, batch.batch)
            target_stress_dense, _ = to_dense_batch(batch.y, batch.batch)
            pred_stress = model(h_dense, pos_dense, mask=mask)
            loss = F.huber_loss(pred_stress, target_stress_dense, delta=0.5)
            val_loss += loss.item() * batch.num_graphs

            # Conversion pour calcul des métriques
            preds_np = pred_stress.view(-1).cpu().numpy()
            targets_np = batch.y.view(-1).cpu().numpy()
            
            # R2 par Graphe (nécessite BATCH_SIZE=1 pour cette implémentation)
            r2_score_graph = r2_score(targets_np, preds_np)
            all_graph_r2s.append(r2_score_graph)

            # Collecte pour métriques globales
            all_predictions_global.append(pred_stress.view(-1))
            all_targets_global.append(batch.y.view(-1))
    
    val_loss /= num_val_graphs
     # --- 1. Calcul des R2 Percentiles (sur la distribution des R2 par graphe) ---
    all_graph_r2s = np.array(all_graph_r2s)
    
    r2_90pct = np.percentile(all_graph_r2s, 90)
    r2_50pct = np.percentile(all_graph_r2s, 50)
    r2_10pct = np.percentile(all_graph_r2s, 10)
    
    # --- 2. Calcul des Métriques Globales (RMSE, MAE) ---
    final_predictions_global = torch.cat(all_predictions_global).cpu().numpy()
    final_targets_global = torch.cat(all_targets_global).cpu().numpy()
    
    # [cite_start]RMSE [cite: 306]
    rmse = np.sqrt(mean_squared_error(final_targets_global, final_predictions_global))
    # [cite_start]MAE [cite: 306]
    mae = mean_absolute_error(final_targets_global, final_predictions_global)
    
    metrics = {
        'R2_90PCT': r2_90pct,
        'R2_50PCT': r2_50pct,
        'R2_10PCT': r2_10pct,
        'RMSE': rmse,
        'MAE': mae
    }
    return val_loss, metrics


def train_model(model, train_loader, val_loader, optimizer, config, 
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
        val_loss, metrics = validate(model, val_loader, config.DEVICE, num_val_graphs)
        val_losses.append(val_loss)

        # Affichage des métriques (R2 percentiles et RMSE/MAE globaux)
        metric_display = (
            f" | R2_90: {metrics['R2_90PCT']:.4f} | R2_50: {metrics['R2_50PCT']:.4f} | R2_10: {metrics['R2_10PCT']:.4f}"
            f" | RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}"
        )
        
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
            print(f"Epoch {epoch}/{config.NUM_EPOCHS} | Train Loss (Huber): {train_loss:.6f} | Val Loss (Huber): {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e} {metric_display}| Time : {time.time()-start_time} ✓ Best!")
        else:
            print(f"Epoch {epoch}/{config.NUM_EPOCHS} | Train Loss (Huber): {train_loss:.6f} | Val Loss (Huber): {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e} {metric_display}| Time : {time.time()-start_time}")
    
    print(f"\n🎯 Best validation loss(huber): {best_val_loss:.6f}")
    print("="*70 + "\n")
    
    return train_losses, val_losses, metrics


def plot_losses(train_losses, val_losses, metrics, save_path):
    """
    Trace et sauvegarde les courbes de loss (Huber).
    """
    best_val = min(val_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss (Huber)", linewidth=2)
    plt.plot(val_losses, label="Validation Loss (Huber)", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Huber)", fontsize=12)
    plt.title("Training & Validation Loss", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.yscale('log')
    # Ajout des métriques (même structure que 'metrics' retourné par train_model)
    if metrics is not None:
        # On suppose que 'metrics' est un dict avec ces clés, sinon on met "N/A"
        r2_90 = metrics.get('R2_90PCT', None)
        r2_50 = metrics.get('R2_50PCT', None)
        r2_10 = metrics.get('R2_10PCT', None)
        rmse = metrics.get('RMSE', None)
        mae = metrics.get('MAE', None)

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else "N/A"

        metrics_text = (
            f"R2_90: {fmt(r2_90)} | "
            f"R2_50: {fmt(r2_50)} | "
            f"R2_10: {fmt(r2_10)} | "
            f"RMSE: {fmt(rmse)} | "
            f"MAE: {fmt(mae)}"
        )

        plt.figtext(0.5, -0.08, metrics_text, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Courbe de loss sauvegardée: {save_path}")
    plt.show()