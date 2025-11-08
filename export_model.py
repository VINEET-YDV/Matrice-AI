import torch
from ultralytics import YOLO

def export_model_to_onnx(model_name='yolov8n.pt', export_name='yolov8n.onnx'):
    """
    Exports a YOLOv8 PyTorch model to the ONNX format.
    
    ONNX (Open Neural Network Exchange) is a high-performance format that
    can be run by various inference engines (like ONNX Runtime).
    
    We enable:
    - half=True: Exports the model in FP16 precision. This is much faster
                 on modern GPUs and uses half the VRAM.
    - dynamic=True: Allows the model to accept dynamic batch sizes (e.g., 1, 4, 8...),
                    which is critical for our dynamic batching strategy.
    """
    print(f"Loading model '{model_name}'...")
    model = YOLO(model_name)
    
    # Define a dummy input shape. 
    dummy_input = torch.randn(1, 3, 640, 640)

    print(f"Exporting model to ONNX format as '{export_name}'...")
    
    try:
        model.export(
            format='onnx',
            imgsz=640,
            half=True,       # Use FP16
            dynamic=True,    # Enable dynamic batch axis
            opset=12         # A widely compatible ONNX opset version
        )
        # Rename for predictable output
        exported_file = model_name.replace('.pt', '.onnx')
        import os
        os.rename(exported_file, export_name)
        
        print(f"\nSuccessfully exported model to '{export_name}'")
        
    except Exception as e:
        print(f"\nAn error occurred during export: {e}")
        print("Please ensure you have 'ultralytics' and 'onnx' installed.")

if __name__ == "__main__":
    export_model_to_onnx(model_name='yolov8n.pt', export_name='yolov8n.onnx')