"""Chargement et préparation des données."""

import torch
import sys
import types
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from utils import UnitGaussianNormalizer


# Hack pour charger les données avec le normalizer
sys.modules['utils'] = types.ModuleType('utils')
sys.modules['utils'].UnitGaussianNormalizer = UnitGaussianNormalizer


class TupleDataset(InMemoryDataset):
    """Dataset PyTorch Geometric à partir de tuples (data, slices)."""
    
    def __init__(self, data, slices):
        super().__init__(None)
        self.data = data
        self.slices = slices

    def len(self):
        return self.data.num_graphs if hasattr(self.data, 'num_graphs') else self.slices['x'].size(0) - 1

    def get(self, idx):
        return self.data.__class__.from_data_list([self.data])[idx]


def get_graph_from_tuple(data, slices, idx):
    """Récupère le idx-ième graphe depuis le tuple (data, slices)."""
    data_dict = {}
    for key in slices.keys():
        start, end = slices[key][idx].item(), slices[key][idx + 1].item()
        
        if key in ['edge_index', 'face'] and data[key].dim() == 2:
            data_dict[key] = data[key][:, start:end]
        else:
            data_dict[key] = data[key][start:end]
    
    #Extract the position from the features
    data_dict['pos'] = data_dict['x'][:, :3]
    data_dict['x'] = data_dict['x'][:, 3:]

    return Data(**data_dict)


def load_data(config):
    """
    Charge les données d'entraînement et de validation.
    
    Args:
        config: Objet de configuration
        
    Returns:
        train_loader, val_loader, num_train_graphs, num_val_graphs
    """
    print("="*70)
    print("CHARGEMENT DES DONNÉES")
    print("="*70)
    
    # Charger train
    train_dataset_tuple = torch.load(config.TRAIN_DATA_PATH)
    train_data, train_slices = train_dataset_tuple
    
    # Réduire le dataset train
    num_graphs = train_slices['x'].size(0) - 1
    keep = int(num_graphs * config.TRAIN_SUBSET_RATIO)
    print(f"\n📊 Dataset train: {num_graphs} graphes → Gardés: {keep} ({config.TRAIN_SUBSET_RATIO*100:.0f}%)")
    
    new_slices = {}
    for key in train_slices.keys():
        new_slices[key] = train_slices[key][:keep+1].clone()
    train_slices = new_slices
    
    # Charger validation
    test_dataset_tuple = torch.load(config.VAL_DATA_PATH)
    test_data, test_slices = test_dataset_tuple
    print(f"📊 Dataset val: {test_slices['x'].size(0) - 1} graphes")
    
    # Créer les graphes
    print("\n🔄 Extraction des graphes...")
    train_graphs = [get_graph_from_tuple(train_data, train_slices, i) 
                    for i in range(train_slices['x'].size(0) - 1)]
    val_graphs = [get_graph_from_tuple(test_data, test_slices, i) 
                  for i in range(test_slices['x'].size(0) - 1)]
    
    print(f"✓ Train: {len(train_graphs)} graphes")
    print(f"✓ Val: {len(val_graphs)} graphes")
    
    # Créer les dataloaders
    train_loader = DataLoader(train_graphs, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print("\n✓ DataLoaders créés")
    print("="*70 + "\n")
    
    return train_loader, val_loader, len(train_graphs), len(val_graphs)