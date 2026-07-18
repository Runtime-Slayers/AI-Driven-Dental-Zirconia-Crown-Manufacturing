import torch
import sys
import os

print("Starting export...")

# 1. Export Segmentation Model
try:
    sys.path.append(os.path.abspath('Stage1_Segmentation'))
    from model import AttentionUNet
    seg_model = AttentionUNet()
    seg_model.eval()
    
    # Export to ONNX for Netron
    dummy_input = torch.randn(1, 1, 256, 256)
    torch.onnx.export(seg_model, dummy_input, 'Stage1_Segmentation/segmentation_unet.onnx', 
                      export_params=True, opset_version=13, 
                      input_names=['input'], output_names=['output'])
    print("Successfully exported Segmentation model to Stage1_Segmentation/segmentation_unet.onnx")
except Exception as e:
    print(f"Error exporting Segmentation model: {e}")

# 2. Export Transformer Model
try:
    sys.path.append(os.path.abspath('Stage2_Transformer'))
    from train_transformer import BidirectionalGraphTransformer
    trans_model = BidirectionalGraphTransformer(D_dim=64, num_heads=4, num_layers=3)
    trans_model.eval()
    
    # Export to ONNX for Netron
    dummy_nodes = torch.randn(1, 150, 3)
    dummy_forces = torch.randn(1, 150, 3)
    torch.onnx.export(trans_model, (dummy_nodes, dummy_forces), 'Stage2_Transformer/transformer_stress.onnx', 
                      export_params=True, opset_version=13, 
                      input_names=['nodes_coord', 'forces'], output_names=['stress_pred'])
    print("Successfully exported Transformer model to Stage2_Transformer/transformer_stress.onnx")
except Exception as e:
    print(f"Error exporting Transformer model: {e}")
