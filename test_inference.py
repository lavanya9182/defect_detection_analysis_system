import torch
from anomalib.deploy import TorchInferencer
import numpy as np
from PIL import Image
import os
import sys

# Allow loading pickle models
os.environ["TRUST_REMOTE_CODE"] = "True"

def test_inference():
    weights_path = "results/weights/weights/torch/model.pt"
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    # Load model
    print("Loading model...")
    inferencer = TorchInferencer(
        path=weights_path,
        device="cpu", 
    )

    # Test on a known defect image
    test_image_path = "dataset/capsule/test/crack/000.png"
    if not os.path.exists(test_image_path):
        # Fallback if specific file doesn't exist, try to find one
        print(f"File {test_image_path} not found. Searching for a test image...")
        # Just pick one from test folder
        for root, dirs, files in os.walk("dataset/capsule/test"):
            for f in files:
                if f.endswith(".png"):
                    test_image_path = os.path.join(root, f)
                    break
            if test_image_path != "dataset/capsule/test/crack/000.png":
                break
    
    print(f"Testing on image: {test_image_path}")
    
    image = np.array(Image.open(test_image_path).convert("RGB"))
    
    # Predict
    print("Running prediction...")
    predictions = inferencer.predict(image=image)
    
    print(f"Prediction Label: {'Defect' if predictions.pred_label else 'Good'}")
    print(f"Anomaly Score: {predictions.pred_score}")
    
    if predictions.pred_label:
        print("Success: Defect detected.")
    else:
        print("Warning: Expected defect, got Good (unless the random image was good).")
        # Note: crack/000.png should be defective.

if __name__ == "__main__":
    test_inference()
