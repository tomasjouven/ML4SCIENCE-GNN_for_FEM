"""Utilitaires : normalizer et helpers."""

import torch


class UnitGaussianNormalizer:
    """Normalisation Gaussienne pour les données."""
    
    def __init__(self, x=None, eps=1e-5):
        self.eps = eps
        if x is not None:
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True) + eps
        else:
            self.mean = None
            self.std = None
    
    def encode(self, x):
        if self.mean is None:
            return x
        return (x - self.mean) / self.std
    
    def decode(self, x, sample_idx=None):
        if self.mean is None:
            return x
        return x * self.std + self.mean