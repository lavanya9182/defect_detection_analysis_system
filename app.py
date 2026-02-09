import streamlit as st
import numpy as np
from PIL import Image
import torch
from anomalib.deploy import TorchInferencer
import cv2
import os
import tempfile
import uuid

# Custom Modules
import db
import analytics
import report

# Allow loading pickle models
os.environ["TRUST_REMOTE_CODE"] = "True"

# Initialize DB
db.init_db()

st.set_page_config(page_title="Pharma QA System", layout="wide", page_icon="💊")

# --- Custom CSS for "Super" UI ---
st.markdown("""
<style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    h1 {
        color: #2E86C1;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #555;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        color: #2E86C1;
        border-bottom: 2px solid #2E86C1;
    }

    /* Button Styling */
    .stButton button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #1B4F72;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("Integrated Pharmaceutical Quality Analysis and Reporting System 💊")

# --- Tabs ---
tab_inference, tab_analytics = st.tabs(["🕵️ Inference & Inspection", "📊 Analytics & Reporting"])

# --- Model Loading ---
@st.cache_resource
def load_models(anomaly_weights_path, classifier_weights_path):
    models = {}
    


    # 1. Anomaly Model (PatchCore)
    # Check for split parts first (GitHub workaround)
    if not os.path.exists(anomaly_weights_path):
        part_prefix = anomaly_weights_path + ".part_"
        # Check manually if parts exist
        if os.path.exists(part_prefix + "aa"): 
            try:
                with open(anomaly_weights_path, 'wb') as dest:
                    for suffix in ['aa', 'ab', 'ac', 'ad']: 
                        part_file = part_prefix + suffix
                        if os.path.exists(part_file):
                            with open(part_file, 'rb') as src:
                                dest.write(src.read())
                print("Model reconstructed successfully!")
            except Exception as e:
                st.error(f"Failed to reconstruct model: {e}")

    if not os.path.exists(anomaly_weights_path):
        st.error(f"Anomaly weights not found at {anomaly_weights_path}. Please run training first.")
    else:
        try:
            models['anomaly'] = TorchInferencer(
                path=anomaly_weights_path,
                device="cpu", 
            )
        except Exception as e:
             st.error(f"Error loading Anomaly Model: {e}")
        
    # 2. Classifier Model (ResNet18)
    if os.path.exists(classifier_weights_path):
        from torchvision import models as tv_models
        from torchvision import transforms
        import torch.nn as nn
        
        try:
            # Load checkpoint
            checkpoint = torch.load(classifier_weights_path, map_location=torch.device('cpu'))
            class_names = checkpoint['class_names']
            
            # Re-create model structure
            classifier = tv_models.resnet18(weights=None)
            num_ftrs = classifier.fc.in_features
            classifier.fc = nn.Linear(num_ftrs, len(class_names))
            
            # Load weights
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier.eval()
            
            models['classifier'] = classifier
            models['classes'] = class_names
            
            # Transforms for classifier
            models['transforms'] = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        except Exception as e:
            st.error(f"Failed to load classifier: {e}")
            st.warning("Continuing without classifier...")
            
    else:
        st.warning(f"Classifier weights not found at {classifier_weights_path}. Automatic defect typing disabled.")
        
    return models

# Paths
ANOMALY_MODEL_PATH = "results/weights/weights/torch/model.pt"
CLASSIFIER_MODEL_PATH = "results/classifier.pth"

loaded_models = load_models(ANOMALY_MODEL_PATH, CLASSIFIER_MODEL_PATH)
model = loaded_models.get('anomaly')
classifier = loaded_models.get('classifier')
class_names = loaded_models.get('classes')
classifier_transforms = loaded_models.get('transforms')

# ==========================================
# TAB 1: INFERENCE
# ==========================================
with tab_inference:
    st.markdown("### Real-time Inspection")
    uploaded_file = st.file_uploader("Upload Capsule Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # Preprocessing
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        if st.button("Analyze Image"):
            if model is None:
                st.error("Model not loaded.")
            else:
                with st.spinner("Analyzing..."):
                    # Inference
                    predictions = model.predict(image=image_np)
                    
                    pred_score = predictions.pred_score
                    if isinstance(pred_score, torch.Tensor):
                        pred_score = pred_score.item()

                    anomaly_map = predictions.anomaly_map
                    
                    # --- Thresholding ---
                    # Good images ~25.7, Defects ~70.8. Setting default to 45.0.
                    threshold = st.slider("Anomaly Threshold", min_value=0.0, max_value=100.0, value=45.0, step=0.5)
                    
                    # Logic
                    is_defect = pred_score > threshold
                    pred_label = "Defect" if is_defect else "Good"
                    
                    # Determine Severity based on Score
                    severity = "Normal"
                    if is_defect:
                        if pred_score < 60:
                            severity = "Minor"
                        elif pred_score < 80:
                            severity = "Medium"
                        else:
                            severity = "Critical"
                            
                    # --- Automatic Defect Classification ---
                    final_defect_type = "Normal"
                    
                    if is_defect:
                        if classifier is not None:
                            # Prepare image for classifier (using PIL image)
                            input_tensor = classifier_transforms(image).unsqueeze(0) # Add batch dim
                            
                            with torch.no_grad():
                                outputs = classifier(input_tensor)
                                _, preds = torch.max(outputs, 1)
                                predicted_class = class_names[preds[0]]
                                
                            # Logic: If classifier says "good" but PatchCore says "Defect",
                            # we report "Unclassified" or potential false positive.
                            if predicted_class == "good":
                                final_defect_type = "Unclassified (Potential False Positive)"
                            else:
                                final_defect_type = predicted_class
                        else:
                             final_defect_type = "General Defect"
                    else:
                        final_defect_type = "Normal (Good)"
                    
                    # --- Results ---
                    st.success("Analysis Complete!")
                    
                    # Log to DB
                    inspection_id = str(uuid.uuid4())[:8]
                    db.log_inspection(
                        image_id=inspection_id,
                        status=pred_label,
                        defect_type=final_defect_type,
                        score=pred_score,
                        severity=severity
                    )
                    st.toast(f"Result logged to database (ID: {inspection_id})")
                    
                    # Metrics Display
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Status", pred_label, delta_color="inverse" if is_defect else "normal")
                    m_col1.metric("Score", f"{pred_score:.2f}")
                    m_col2.metric("Severity", severity)
                    m_col3.metric("Defect Type", final_defect_type)

                    # Visualization
                    with col2:
                        st.subheader("Segmentation Heatmap")
                        
                        if isinstance(anomaly_map, torch.Tensor):
                             anomaly_map = anomaly_map.cpu().numpy()
                        
                        if anomaly_map.ndim == 3 and anomaly_map.shape[0] == 1:
                            anomaly_map = anomaly_map.squeeze(0)
                        
                        am_min, am_max = anomaly_map.min(), anomaly_map.max()
                        if am_max > am_min:
                            heatmap_norm = (anomaly_map - am_min) / (am_max - am_min)
                        else:
                            heatmap_norm = anomaly_map
                            
                        heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                        
                        st.image(heatmap_color, caption="Anomaly Heatmap", use_column_width=True)

# ==========================================
# TAB 2: ANALYTICS
# ==========================================
with tab_analytics:
    st.markdown("### 📊 QA Analytics Dashboard")
    
    # Filters
    period = st.selectbox("Select Time Period", ["All", "Daily", "Weekly", "Monthly"])
    
    if st.button("Refresh Data"):
        st.cache_data.clear()

    # Fetch Data
    df = db.fetch_data(period)
    
    if df.empty:
        st.info("No inspection data found for this period.")
    else:
        # Calculate Stats
        stats = analytics.calculate_stats(df)
        
        # Summary Metrics
        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        a_col1.metric("Total Inspected", stats['total'])
        a_col2.metric("Defect Rate", f"{stats['defect_rate']:.1f}%")
        a_col3.metric("Defects Found", stats['defect_count'])
        a_col4.metric("Most Frequent", stats['most_frequent_defect'])
        
        st.markdown("---")
        
        # Charts
        c_col1, c_col2 = st.columns(2)
        
        chart_dist = analytics.get_defect_distribution_chart(df)
        chart_status = analytics.get_status_distribution_chart(df)
        chart_trend = analytics.get_trend_chart(df)
        
        with c_col1:
            if chart_status: st.pyplot(chart_status)
        with c_col2:
            if chart_dist: st.pyplot(chart_dist)
            
        st.markdown("#### Defect Trends")
        if chart_trend: st.pyplot(chart_trend)
        
        st.markdown("---")
        st.subheader("📄 Generate Report")
        
        if st.button("Generate PDF Report"):
            charts = [chart_status, chart_dist, chart_trend]
            df_defects = df[df['status'] == 'Defect']
            
            report_file = report.generate_qa_report(period, stats, df_defects, charts)
            
            with open(report_file, "rb") as f:
                pdf_bytes = f.read()
                
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=report_file,
                mime="application/pdf"
            )