import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import AttentionUNet

# Custom Dataset
class DentalCBCTDataset(Dataset):
    def __init__(self, preprocessed_dir):
        self.files = sorted(glob.glob(os.path.join(preprocessed_dir, "*.npz")))
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        # Raw image normalized to [0, 1]
        raw = data['raw'].astype(np.float32) / 255.0
        mask = data['mask'].astype(np.float32)
        
        # Add channel dimension
        raw = np.expand_dims(raw, axis=0)  # Shape (1, H, W)
        mask = np.expand_dims(mask, axis=0) # Shape (1, H, W)
        
        return torch.from_numpy(raw), torch.from_numpy(mask)

# Focal + Dice Loss
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, smooth=1e-6):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, pred_logits, targets):
        pred_probs = torch.sigmoid(pred_logits)
        
        # 1. Focal Loss
        bce = F_loss = nn.functional.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt)**self.gamma * bce
        focal_loss = focal.mean()
        
        # 2. Dice Loss
        intersection = (pred_probs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred_probs.sum() + targets.sum() + self.smooth)
        
        return focal_loss + dice_loss

def train_model(preprocessed_dir, model_save_path, epochs=3, batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = DentalCBCTDataset(preprocessed_dir)
    # Split into train/val
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = AttentionUNet(in_ch=1, out_ch=1, dropout_p=0.2).to(device)
    criterion = FocalDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print("Saved best model weights.")

def mc_dropout_inference(model_path, sample_npz_path, num_samples=50):
    """
    Runs Monte Carlo Dropout inference on a single sample to generate boundary uncertainty maps.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AttentionUNet(in_ch=1, out_ch=1, dropout_p=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Force dropout layers to remain active during evaluation
    model.train() # Keeping the model in train mode enables dropout at inference
    
    data = np.load(sample_npz_path)
    raw = data['raw'].astype(np.float32) / 255.0
    raw = np.expand_dims(raw, axis=(0, 1)) # Shape (1, 1, H, W)
    raw_tensor = torch.from_numpy(raw).to(device)
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = torch.sigmoid(model(raw_tensor))
            predictions.append(outputs.cpu().numpy()[0, 0])
            
    predictions = np.array(predictions) # Shape (num_samples, H, W)
    
    # Calculate mean and variance
    mean_pred = np.mean(predictions, axis=0)
    var_pred = np.var(predictions, axis=0) # Epistemic uncertainty map
    
    print(f"MC Dropout complete. Variance range: {var_pred.min():.6f} to {var_pred.max():.6f}")
    return mean_pred, var_pred

if __name__ == "__main__":
    preprocessed_dir = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage1_Segmentation/preprocessed"
    model_save_path = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage1_Segmentation/best_unet.pth"
    
    # Run short training for baseline validation
    train_model(preprocessed_dir, model_save_path, epochs=3, batch_size=4)
    
    # Test MC Dropout on the first sample
    first_file = sorted(glob.glob(os.path.join(preprocessed_dir, "*.npz")))[0]
    mean, var = mc_dropout_inference(model_save_path, first_file, num_samples=10)
    
    # Save the output maps
    np.savez_compressed(
        "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage1_Segmentation/mc_uncertainty_sample.npz",
        mean=mean,
        variance=var
    )
    print("Saved MC Dropout uncertainty maps successfully!")
