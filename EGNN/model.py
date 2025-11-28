"""Définition de l'architecture du modèle."""

import torch
import torch.nn as nn
from egnn_pytorch import EGNN


class EGNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Linear(config.IN_NODE_FEATURES, config.HIDDEN_CHANNELS)

        self.stress_head = nn.Sequential(
            nn.Linear(config.HIDDEN_CHANNELS, config.HIDDEN_CHANNELS),
            nn.ReLU(), 
            nn.Linear(config.HIDDEN_CHANNELS, config.OUT_STRESS_DIM)
        )
        
        self.layers = nn.ModuleList([])
        for _ in range(config.DEPTH):
            self.layers.append(EGNN(
                dim=config.HIDDEN_CHANNELS,      # Size of h
                m_dim=config.HIDDEN_CHANNELS,    # Size of message
                num_nearest_neighbors=config.NUM_NEIGHBORS,
                update_coors=config.UPDATE_COORS, # Critical for deformation
                update_feats=config.UPDATE_FEATS,
                norm_coors=True,               
                soft_edges=True                
            ))

    def forward(self, h, pos, mask=None):
            h = self.embedding(h)

            for layer in self.layers:
                h, _ = layer(h, pos, mask=mask)

            pred_stress = self.stress_head(h) 
            return pred_stress

def create_model(config):
    model = EGNNModel(config)
    return model.to(config.DEVICE)