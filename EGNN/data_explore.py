import torch
import types
import sys
from torch_geometric.data import Data
from torch_geometric.nn import GraphUNet
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import FaceToEdge

batch_size = 1  # Ajuster selon la RAM

# Normalizer
class UnitGaussianNormalizer:
    def __init__(self, x=None, eps=1e-5):
        self.eps = eps
        if x is not None:
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True) + eps
        else:
            self.mean = None
            self.std = None
    
    def encode(self, x):
        if self.mean is None: return x
        return (x - self.mean) / self.std
    
    def decode(self, x, sample_idx=None):
        if self.mean is None: return x
        return x * self.std + self.mean

sys.modules['utils'] = types.ModuleType('utils')
sys.modules['utils'].UnitGaussianNormalizer = UnitGaussianNormalizer

# Simplified graph extraction
def get_graph_from_tuple(data, slices, idx):
    """Récupère le idx-ième graphe depuis le tuple (data, slices)."""
    # Garde uniquement x, y, y_original et normalizer
    data_dict = {
        'x': data.x[slices['x'][idx]:slices['x'][idx+1]],
        'y': data.y[slices['y'][idx]:slices['y'][idx+1]],
        'y_original': data.y_original[slices['y_original'][idx]:slices['y_original'][idx+1]],
        'normalizer': data.normalizer
    }
    return Data(**data_dict)

# LOADING
train_data, train_slices = torch.load('data/train_data.pt')
train_graphs = [get_graph_from_tuple(train_data, train_slices, i) 
                for i in range(train_slices['x'].size(0) - 1)]

val_data, val_slices = torch.load('data/val_data.pt')
val_graphs = [get_graph_from_tuple(val_data, val_slices, i) 
              for i in range(val_slices['x'].size(0) - 1)]

# DataLoader
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

# Exemple
print("x shape:", train_graphs[0].keys())
