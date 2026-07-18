import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. Twin Reservoir Encoder
class TwinReservoirEncoder(nn.Module):
    def __init__(self, R_dim=256, D_dim=64):
        super(TwinReservoirEncoder, self).__init__()
        self.R_dim = R_dim
        self.D_dim = D_dim
        
        # Fixed random projections (Reservoirs)
        self.register_buffer("W_geo", torch.randn(3, R_dim))
        self.register_buffer("W_load", torch.randn(3, R_dim))
        
        # Trainable Autoencoder compression layers
        self.compress_geo = nn.Sequential(
            nn.Linear(R_dim, D_dim),
            nn.ReLU(),
            nn.Linear(D_dim, D_dim)
        )
        self.compress_load = nn.Sequential(
            nn.Linear(R_dim, D_dim),
            nn.ReLU(),
            nn.Linear(D_dim, D_dim)
        )
        
    def forward(self, nodes_coord, forces):
        # inputs are shape (batch_size, num_nodes, 3)
        # 1. Compute high-dimensional reservoir activations
        # X_geo = tanh(coord * W_geo)
        h_geo = torch.tanh(torch.matmul(nodes_coord, self.W_geo))
        # X_load = tanh(forces * W_load)
        h_load = torch.tanh(torch.matmul(forces, self.W_load))
        
        # 2. Compress via trainable autoencoder layers
        z_geo = self.compress_geo(h_geo)
        z_load = self.compress_load(h_load)
        
        # Combine representations (addition or concatenation)
        return z_geo + z_load

# 2. Bidirectional Graph Transformer
class BidirectionalGraphTransformer(nn.Module):
    def __init__(self, D_dim=64, num_heads=4, num_layers=3, num_nodes=150):
        super(BidirectionalGraphTransformer, self).__init__()
        
        self.reservoir_encoder = TwinReservoirEncoder(R_dim=256, D_dim=D_dim)
        
        # 3D Positional Embedding for node coordinates
        self.pos_embedding = nn.Linear(3, D_dim)
        
        # Bidirectional Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_dim,
            nhead=num_heads,
            dim_feedforward=2*D_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output stress regression head
        self.stress_head = nn.Sequential(
            nn.Linear(D_dim, D_dim // 2),
            nn.ReLU(),
            nn.Linear(D_dim // 2, 1)
        )
        
    def forward(self, nodes_coord, forces):
        # 1. Twin Reservoir representation
        z_combined = self.reservoir_encoder(nodes_coord, forces)
        
        # 2. Add spatial positional embeddings
        pos = self.pos_embedding(nodes_coord)
        x = z_combined + pos
        
        # 3. Propagate boundary information bidirectionally
        features = self.transformer_encoder(x)
        
        # 4. Predict nodal stress
        stress_pred = self.stress_head(features) # Shape (batch_size, num_nodes, 1)
        return stress_pred.squeeze(-1)

# Custom Dataset
class BiomechanicalFEMDataset(Dataset):
    def __init__(self, dataset_dir):
        self.files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
        print(f"Loaded {len(self.files)} patient FEM samples.")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        nodes = data['nodes'].astype(np.float32)      # (150, 3)
        forces = data['forces'].astype(np.float32)    # (150, 3)
        von_mises = data['von_mises'].astype(np.float32) # (150,)
        
        return torch.from_numpy(nodes), torch.from_numpy(forces), torch.from_numpy(von_mises)

def train_transformer(dataset_dir, model_save_path, epochs=30, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = BiomechanicalFEMDataset(dataset_dir)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = BidirectionalGraphTransformer(D_dim=64, num_heads=4, num_layers=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for nodes, forces, targets in train_loader:
            nodes, forces, targets = nodes.to(device), forces.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(nodes, forces)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * nodes.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for nodes, forces, targets in val_loader:
                nodes, forces, targets = nodes.to(device), forces.to(device), targets.to(device)
                outputs = model(nodes, forces)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * nodes.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print("Saved best transformer weights.")

if __name__ == "__main__":
    # Local path for generating/training
    dataset_dir = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage2_Transformer/fem_dataset"
    model_save_path = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage2_Transformer/transformer_stress.pth"
    
    # If running on Kaggle, paths adjust automatically
    if os.path.exists("/kaggle/input/biomechanical-fem-dataset"):
        dataset_dir = "/kaggle/input/biomechanical-fem-dataset"
        model_save_path = "transformer_stress.pth"
        
    train_transformer(dataset_dir, model_save_path, epochs=30, batch_size=8)
    print("Transformer training execution complete.")
