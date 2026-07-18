import os
import sys
import subprocess

# Self-healing PyTorch reinstallation for Tesla P100 (sm_60) compatibility
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 7:
        print("Detected older GPU (capability < 7.0). Reinstalling official PyTorch with sm_60 support...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "torch==2.3.0+cu121", "torchvision==0.18.0+cu121", "--extra-index-url", "https://download.pytorch.org/whl/cu121"])
        import importlib
        importlib.reload(torch)
        print("PyTorch reloaded. New version:", torch.__version__)
except Exception as e:
    print("Self-healing PyTorch check skipped/failed:", str(e))

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Attention Gate
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g_in = self.W_g(g)
        x_in = self.W_x(x)
        
        if g_in.shape[-2:] != x_in.shape[-2:]:
            g_in = F.interpolate(g_in, size=x_in.shape[-2:], mode='bilinear', align_corners=True)
            
        psi = self.relu(g_in + x_in)
        psi = self.psi(psi)
        
        return x * psi

# Conv Block
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dropout_p=0.0):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv(x)

# Attention U-Net
class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, dropout_p=0.2):
        super(AttentionUNet, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = ConvBlock(ch_in=in_ch, ch_out=64, dropout_p=dropout_p)
        self.Conv2 = ConvBlock(ch_in=64, ch_out=128, dropout_p=dropout_p)
        self.Conv3 = ConvBlock(ch_in=128, ch_out=256, dropout_p=dropout_p)
        self.Conv4 = ConvBlock(ch_in=256, ch_out=512, dropout_p=dropout_p)
        
        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.Att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = ConvBlock(ch_in=512, ch_out=256, dropout_p=dropout_p)
        
        self.Up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = ConvBlock(ch_in=256, ch_out=128, dropout_p=dropout_p)
        
        self.Up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.Att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = ConvBlock(ch_in=128, ch_out=64, dropout_p=dropout_p)
        
        self.Conv_1x1 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        d4 = self.Up4(x4)
        x3_att = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        x2_att = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        x1_att = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        out = self.Conv_1x1(d2)
        
        return out

# Custom Dataset
class DentalCBCTDataset(Dataset):
    def __init__(self, preprocessed_dir):
        # Scan for npz files
        self.files = sorted(glob.glob(os.path.join(preprocessed_dir, "*.npz")))
        print(f"Dataset initialized with {len(self.files)} preprocessed samples.")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        raw = data['raw'].astype(np.float32) / 255.0
        mask = data['mask'].astype(np.float32)
        
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
        
        bce = nn.functional.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt)**self.gamma * bce
        focal_loss = focal.mean()
        
        intersection = (pred_probs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred_probs.sum() + targets.sum() + self.smooth)
        
        return focal_loss + dice_loss

def train_model(preprocessed_dir, model_save_path, epochs=15, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = DentalCBCTDataset(preprocessed_dir)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AttentionUNet(in_ch=1, out_ch=1, dropout_p=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.train() # Enable dropout at inference
    
    data = np.load(sample_npz_path)
    raw = data['raw'].astype(np.float32) / 255.0
    raw = np.expand_dims(raw, axis=(0, 1))
    raw_tensor = torch.from_numpy(raw).to(device)
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = torch.sigmoid(model(raw_tensor))
            predictions.append(outputs.cpu().numpy()[0, 0])
            
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    var_pred = np.var(predictions, axis=0)
    
    print(f"MC Dropout complete. Variance range: {var_pred.min():.6f} to {var_pred.max():.6f}")
    return mean_pred, var_pred

if __name__ == "__main__":
    import zipfile
    
    # 1. Search for npz files first (if Kaggle unzipped it automatically)
    npz_dir = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if any(f.endswith('.npz') for f in files):
            npz_dir = root
            break
            
    if npz_dir is not None:
        print(f"Found unzipped npz files at: {npz_dir}")
        extract_dir = npz_dir
    else:
        # 2. Search for zip files to extract
        zip_path = None
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if f.endswith('.zip'):
                    zip_path = os.path.join(root, f)
                    break
            if zip_path:
                break
                
        if zip_path is not None:
            extract_dir = "/kaggle/working/preprocessed"
            os.makedirs(extract_dir, exist_ok=True)
            print(f"Unzipping {zip_path} to {extract_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("Unzip complete.")
        else:
            raise FileNotFoundError("Could not find any npz or zip files in /kaggle/input")
            
    model_save_path = "best_unet.pth"
    train_model(extract_dir, model_save_path, epochs=15, batch_size=8)
    
    # Run MC Dropout on the first sample to generate uncertainty outputs
    first_file = sorted(glob.glob(os.path.join(extract_dir, "*.npz")))[0]
    mean, var = mc_dropout_inference(model_save_path, first_file, num_samples=30)
    
    np.savez_compressed("mc_uncertainty_sample.npz", mean=mean, variance=var)
    print("Execution complete. Output files saved in working directory.")
