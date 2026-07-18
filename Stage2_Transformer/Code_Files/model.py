import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGEDownsample(nn.Module):
    """
    Downsamples a variable-sized mesh to a fixed number of super-nodes (e.g., 150)
    to allow the transformer to handle real patient meshes instead of being capped.
    """
    def __init__(self, in_dim, out_dim, num_super_nodes=150):
        super().__init__()
        self.num_super_nodes = num_super_nodes
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        
    def forward(self, x):
        # x shape: (B, N, in_dim)
        B, N, _ = x.shape
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        # Simple adaptive pooling to act as super-node downsampling
        # In a real PyTorch Geometric implementation, this would be TopKPooling or EdgePooling
        x = x.transpose(1, 2) # (B, out_dim, N)
        x = F.adaptive_max_pool1d(x, self.num_super_nodes) # (B, out_dim, num_super_nodes)
        x = x.transpose(1, 2) # (B, num_super_nodes, out_dim)
        return x

class TwinReservoirEncoder(nn.Module):
    def __init__(self, d_dim=64):
        super().__init__()
        self.d_dim = d_dim
        # tanh(W*x)
        self.coord_w = nn.Linear(3, 32)
        self.force_w = nn.Linear(3, 32)
        
        self.coord_ae = nn.Linear(32, d_dim)
        self.force_ae = nn.Linear(32, d_dim)
        
    def forward(self, coords, forces):
        c = torch.tanh(self.coord_w(coords))
        f = torch.tanh(self.force_w(forces))
        
        c = self.coord_ae(c)
        f = self.force_ae(f)
        
        # Fusion
        return c + f

class FourierPositionalEmbedding(nn.Module):
    def __init__(self, d_dim=64):
        super().__init__()
        self.d_dim = d_dim
        self.B = nn.Parameter(torch.randn(3, d_dim // 2))
        
    def forward(self, coords):
        # coords: (B, N, 3)
        proj = 2 * torch.pi * coords @ self.B
        emb = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return emb

class TwinReservoirBiTransformer(nn.Module):
    """
    Twin-Reservoir Bidirectional Transformer (Graph-Augmented).
    Predicts von Mises stress from node coordinates and forces.
    """
    def __init__(self, d_dim=64, num_heads=4, num_layers=3, num_super_nodes=150):
        super().__init__()
        self.encoder = TwinReservoirEncoder(d_dim)
        
        # GraphSAGE Downsampler to fix the 150-node assumption on real meshes
        self.downsample = GraphSAGEDownsample(d_dim, d_dim, num_super_nodes)
        self.coord_downsample = GraphSAGEDownsample(3, 3, num_super_nodes)
        
        self.pos_emb = FourierPositionalEmbedding(d_dim)
        
        # Graph Attention based Transformer (using standard TransformerEncoder for demo, 
        # but configured to act as full-graph attention on the supernodes)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_dim, nhead=num_heads, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression head for von Mises stress
        self.regression = nn.Sequential(
            nn.Linear(d_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output: von Mises stress
        )
        
    def forward(self, coords, forces):
        # coords: (B, N, 3), forces: (B, N, 3)
        # 1. Twin Reservoir Encoding
        features = self.encoder(coords, forces) # (B, N, d_dim)
        
        # 2. GraphSAGE Downsampling to super-nodes (solves Defect S2-2)
        features_sn = self.downsample(features) # (B, 150, d_dim)
        coords_sn = self.coord_downsample(coords) # (B, 150, 3)
        
        # 3. Positional Embedding
        pos = self.pos_emb(coords_sn)
        x = features_sn + pos
        
        # 4. Bidirectional Transformer (Attention over graph)
        x = self.transformer(x) # (B, 150, d_dim)
        
        # 5. Regression
        stress_sn = self.regression(x) # (B, 150, 1)
        
        # Upsample back to N nodes via simple interpolation/broadcasting
        # In practice, use nearest neighbor from coords_sn to coords
        B, N, _ = coords.shape
        # Simple dummy upsample for shape matching
        stress = F.interpolate(stress_sn.transpose(1, 2), size=N, mode='nearest').transpose(1, 2)
        return stress

def physics_equilibrium_loss(predicted_stress, coords, forces):
    """
    Computes a soft physics constraint: nabla sigma + b = 0.
    Since predicted_stress here is just a scalar (von Mises), 
    this is a simplified proxy for the full tensor equilibrium.
    """
    # Dummy gradient approximation for demonstration
    # In a real PINN, we use torch.autograd.grad(predicted_stress, coords)
    loss = torch.mean((predicted_stress * 0.01 - forces.norm(dim=-1, keepdim=True))**2)
    return loss
