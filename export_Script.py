import torch
import torch.onnx
from demo.net import build_model # This imports the IResNet structure

def export_to_onnx(checkpoint_path, output_name="adaface_architecture.onnx"):
    # 1. Initialize the skeleton (Backbone)
    # 'ir_101' is the deep version used in your VMS project
    model = build_model('ir_101')
    
    # 2. Load the weights
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # 3. Clean the weights (Remove the 'Head' and 'model.' prefixes)
    # This prevents the "Missing Keys" error from crashing the export
    model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('model.', '') # Remove prefix if it exists
        if name in model_dict and v.size() == model_dict[name].size():
            new_state_dict[name] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("Model loaded successfully.")

    # 4. Create dummy input (Standard InsightFace size: 112x112)
    dummy_input = torch.randn(1, 3, 112, 112)

    # 5. Export
    print(f"Exporting to {output_name}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_name, 
        export_params=True, 
        opset_version=12, 
        do_constant_folding=True, 
        input_names=['input_face'], 
        output_names=['face_embedding'],
        dynamic_axes={'input_face': {0: 'batch_size'}, 'face_embedding': {0: 'batch_size'}}
    )
    print("Success! Now drag the file into Netron.app")

if __name__ == "__main__":
    export_to_onnx('adaface_ir101_ms1mv2.ckpt')