import streamlit as st
import numpy as np
from PIL import Image
import torch
from anomalib.deploy import TorchInferencer
import cv2
import os
import tempfile

# Allow loading pickle models
os.environ["TRUST_REMOTE_CODE"] = "True"

st.set_page_config(page_title="Capsule Defect Detection", layout="wide")

st.title("💊 Capsule Defect Detection")
st.markdown("Upload a capsule image to detect anomalies using the **PatchCore** model.")

# --- Model Loading ---
@st.cache_resource
def load_model(weights_path):
    if not os.path.exists(weights_path):
        st.error(f"Model weights not found at {weights_path}. Please run training first.")
        return None
    
    # Initialize TorchInferencer
    # Note: Anomalib's TorchInferencer expects the path to the torch script or traced model usually, 
    # but let's see if we can load the standard exported model.
    # If training generates a .pt file, we use that.
    inferencer = TorchInferencer(
        path=weights_path,
        device="cpu", # Use CPU for inference on Streamlit unless GPU is available and needed
    )
    return inferencer

# Path where train.py exports the model
MODEL_PATH = "results/weights/weights/torch/model.pt"

model = load_model(MODEL_PATH)

# --- UI ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Basic Preprocessing
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    if st.button("Analyze Image"):
        if model is None:
            st.error("Model not loaded.")
        else:
            with st.spinner("Analyzing..."):
                # Inference
                # TorchInferencer.predict expects numpy array (H, W, C) usually
                predictions = model.predict(image=image_np)
                
                # predictions object usually contains:
                # - pred_score
                # - pred_label
                # - anomaly_map
                # - pred_mask
                
                pred_score = predictions.pred_score
                # Convert tensor to float if necessary
                if isinstance(pred_score, torch.Tensor):
                    pred_score = pred_score.item()

                anomaly_map = predictions.anomaly_map
                
                # --- Thresholding ---
                # Since validation was skipped, the automatic threshold might be inaccurate.
                # PatchCore scores are distances, often > 1.0. 
                # Good images ~25, Defects ~70. Setting default to 45.0.
                threshold = st.slider("Anomaly Threshold (Adjust to filter noise)", min_value=0.0, max_value=100.0, value=45.0, step=0.5)
                
                # Determine label based on manual threshold
                is_defect = pred_score > threshold
                pred_label = "Defect" if is_defect else "Good"
                
                # --- Results ---
                st.success("Analysis Complete!")
                
                # Metrics
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Status", pred_label, delta_color="inverse" if pred_label == "Defect" else "normal")
                m_col1.metric("Anomaly Score", f"{pred_score:.4f}")

                # Visualization
                with col2:
                    st.subheader("Segmentation Result")
                    
                    # Create mask based on threshold from anomaly_map
                    if isinstance(anomaly_map, torch.Tensor):
                         anomaly_map = anomaly_map.cpu().numpy()
                    
                    # Fix shape: (1, 256, 256) -> (256, 256)
                    if anomaly_map.ndim == 3 and anomaly_map.shape[0] == 1:
                        anomaly_map = anomaly_map.squeeze(0)
                    
                    # Normalize heatmap for display 0-1
                    am_min, am_max = anomaly_map.min(), anomaly_map.max()
                    if am_max > am_min:
                        heatmap_norm = (anomaly_map - am_min) / (am_max - am_min)
                    else:
                        heatmap_norm = anomaly_map
                        
                    heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    
                    # Convert BGR to RGB for Streamlit
                    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                    
                    st.image(heatmap_color, caption="Anomaly Heatmap (Color)", use_column_width=True)


