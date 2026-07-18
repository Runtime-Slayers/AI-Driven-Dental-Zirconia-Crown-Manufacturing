import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g_in = self.W_g(g)
        x_in = self.W_x(x)
        
        if g_in.shape[-3:] != x_in.shape[-3:]:
            g_in = F.interpolate(g_in, size=x_in.shape[-3:], mode='trilinear', align_corners=True)
            
        psi = self.relu(g_in + x_in)
        psi = self.psi(psi)
        
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dropout_p=0.0):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv3d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout3d(dropout_p))
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, dropout_p=0.2):
        super(AttentionUNet, self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.Conv1 = ConvBlock(ch_in=in_ch, ch_out=64, dropout_p=dropout_p)
        self.Conv2 = ConvBlock(ch_in=64, ch_out=128, dropout_p=dropout_p)
        self.Conv3 = ConvBlock(ch_in=128, ch_out=256, dropout_p=dropout_p)
        self.Conv4 = ConvBlock(ch_in=256, ch_out=512, dropout_p=dropout_p)
        
        self.Up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.Att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = ConvBlock(ch_in=512, ch_out=256, dropout_p=dropout_p)
        
        self.Up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.Att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = ConvBlock(ch_in=256, ch_out=128, dropout_p=dropout_p)
        
        self.Up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.Att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = ConvBlock(ch_in=128, ch_out=64, dropout_p=dropout_p)
        
        self.Conv_1x1 = nn.Conv3d(64, out_ch, kernel_size=1, stride=1, padding=0)
        
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
