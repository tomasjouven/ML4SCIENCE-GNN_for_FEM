"""Définition de l'architecture du modèle."""

from torch_geometric.nn import GraphUNet


def create_model(config):
    """
    Crée le modèle GraphUNet.
    
    Args:
        config: Objet de configuration
        
    Returns:
        model: Modèle GraphUNet
    """
    model = GraphUNet(
        in_channels=config.IN_CHANNELS,
        hidden_channels=config.HIDDEN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        depth=config.DEPTH,
        pool_ratios=config.POOL_RATIOS,
        sum_res=True,
        act='relu'
    )
    
    model = model.to(config.DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Device: {config.DEVICE}\n")
    
    return model