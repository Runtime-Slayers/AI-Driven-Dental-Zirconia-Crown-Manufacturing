import torch
import torch.nn as nn
from model import TwinReservoirBiTransformer, physics_equilibrium_loss

def train_transformer():
    print("Initializing Twin-Reservoir Bidirectional Transformer...")
    model = TwinReservoirBiTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Mock data for demonstration: Batch size 2, 500 nodes per mesh
    # This proves the model can handle >150 nodes and dynamic downsampling
    coords = torch.rand((2, 500, 3))
    forces = torch.rand((2, 500, 3))
    true_stress = torch.rand((2, 500, 1))
    
    epochs = 100
    print(f"Training on mock FEM dataset (Nodes=500, SuperNodes=150) for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        predicted_stress = model(coords, forces)
        
        # 1. Standard MSE loss against ground truth stress
        mse_loss = nn.MSELoss()(predicted_stress, true_stress)
        
        # 2. Physics-informed equilibrium loss (Defect S2-4)
        phys_loss = physics_equilibrium_loss(predicted_stress, coords, forces)
        
        total_loss = mse_loss + 0.1 * phys_loss
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Total Loss: {total_loss.item():.4f} (MSE: {mse_loss.item():.4f}, Physics: {phys_loss.item():.4f})")
            
    print("Training Complete. Model is ready for inference!")

if __name__ == "__main__":
    train_transformer()
