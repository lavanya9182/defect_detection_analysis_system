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

# Professional UI Theme & CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="st-at"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Container Glassmorphism */
    .stApp {
        background: radial-gradient(circle at top left, #1a1c2c, #0d0e1a);
        color: #ffffff;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(20, 22, 39, 0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Professional Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        transition: transform 0.3s ease;
        margin-bottom: 15px;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
    }
    .metric-label {
        font-size: 14px;
        color: #a0a0c0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Custom Title */
    .dashboard-title {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
        text-align: left;
    }

    /* Custom Subheaders */
    .custom-subheader {
        font-size: 20px;
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #6366f1;
        padding-left: 15px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
        transform: scale(1.02);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>💊</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #6366f1;'>Pharma QA Engine</h3>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ["🕵️ Real-time Inspection", "📊 Analytics Dashboard", "📝 System Info"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**System Health**")
    st.success("Operational")
    st.markdown("---")
    st.caption("v1.5.0 Professional Edition")

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
# PAGE 1: INFERENCE
# ==========================================
if page == "🕵️ Real-time Inspection":
    st.markdown("<h1 class='dashboard-title'>Neural Inspection Interface</h1>", unsafe_allow_html=True)
    
    col_input, col_output = st.columns([1, 1.2])
    
    with col_input:
        st.markdown("<div class='custom-subheader'>Image Acquisition</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Capsule Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            st.image(image, caption="Original Stream", use_column_width=True)
            
            if st.button("🚀 Analyze Capsule"):
                if model is None:
                    st.error("Inference module not available.")
                else:
                    with st.spinner("Processing Semantic Features..."):
                        # Inference Logic
                        predictions = model.predict(image=image_np)
                        pred_score = predictions.pred_score
                        if isinstance(pred_score, torch.Tensor): pred_score = pred_score.item()
                        anomaly_map = predictions.anomaly_map
                        
                        # Thresholding
                        is_defect = pred_score > 45.0
                        pred_label = "Defect" if is_defect else "Good"
                        
                        # Severity
                        severity = "Normal"
                        if is_defect:
                            if pred_score < 60: severity = "Minor"
                            elif pred_score < 80: severity = "Medium"
                            else: severity = "Critical"
                                
                        # Classifier
                        final_defect_type = "Normal"
                        if is_defect:
                            if classifier is not None:
                                input_tensor = classifier_transforms(image).unsqueeze(0)
                                with torch.no_grad():
                                    outputs = classifier(input_tensor)
                                    _, preds = torch.max(outputs, 1)
                                    predicted_class = class_names[preds[0]]
                                    final_defect_type = predicted_class if predicted_class != "good" else "Unclassified"
                            else: final_defect_type = "General Defect"
                        else: final_defect_type = "Normal (Good)"
                        
                        # Database Logging
                        inspection_id = str(uuid.uuid4())[:8]
                        db.log_inspection(inspection_id, pred_label, final_defect_type, pred_score, severity)
                        
                        # Display Results in col_output
                        with col_output:
                            st.markdown("<div class='custom-subheader'>Inspection Summary</div>", unsafe_allow_html=True)
                            
                            # Metric Grid
                            m1, m2 = st.columns(2)
                            with m1:
                                st.markdown(f"""<div class='metric-card'><div class='metric-label'>Status</div><div class='metric-value' style='color: {'#ef4444' if is_defect else '#10b981'};'>{pred_label}</div></div>""", unsafe_allow_html=True)
                                st.markdown(f"""<div class='metric-card'><div class='metric-label'>Severity</div><div class='metric-value'>{severity}</div></div>""", unsafe_allow_html=True)
                            with m2:
                                st.markdown(f"""<div class='metric-card'><div class='metric-label'>Anomaly Score</div><div class='metric-value'>{pred_score:.1f}</div></div>""", unsafe_allow_html=True)
                                st.markdown(f"""<div class='metric-card'><div class='metric-label'>Classification</div><div class='metric-value'>{final_defect_type}</div></div>""", unsafe_allow_html=True)
                            
                            st.markdown("<div class='custom-subheader'>Anomaly Heatmap</div>", unsafe_allow_html=True)
                            if isinstance(anomaly_map, torch.Tensor): anomaly_map = anomaly_map.cpu().numpy()
                            if anomaly_map.ndim == 3: anomaly_map = anomaly_map.squeeze(0)
                            
                            am_min, am_max = anomaly_map.min(), anomaly_map.max()
                            heatmap_norm = (anomaly_map - am_min) / (am_max - am_min) if am_max > am_min else anomaly_map
                            heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                            st.image(heatmap_color, caption="Localized Defects", use_column_width=True)

# ==========================================
# PAGE 2: ANALYTICS
# ==========================================
elif page == "📊 Analytics Dashboard":
    st.markdown("<h1 class='dashboard-title'>Product Quality Intelligence</h1>", unsafe_allow_html=True)
    
    # Controls
    ctrl1, ctrl2 = st.columns([1, 1])
    with ctrl1:
        period = st.selectbox("Historical Window", ["All", "Daily", "Weekly", "Monthly"])
    with ctrl2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data Pipeline"): st.cache_data.clear()

    df = db.fetch_data(period)
    
    if df.empty:
        st.info("No data available for the selected period.")
    else:
        stats = analytics.calculate_stats(df)
        
        # Dashboard Overview Metrics
        o1, o2, o3, o4 = st.columns(4)
        with o1: st.markdown(f"""<div class='metric-card'><div class='metric-label'>Total Inspected</div><div class='metric-value'>{stats['total']}</div></div>""", unsafe_allow_html=True)
        with o2: st.markdown(f"""<div class='metric-card'><div class='metric-label'>Defect Rate</div><div class='metric-value'>{stats['defect_rate']:.1f}%</div></div>""", unsafe_allow_html=True)
        with o3: st.markdown(f"""<div class='metric-card'><div class='metric-label'>Total Defects</div><div class='metric-value'>{stats['defect_count']}</div></div>""", unsafe_allow_html=True)
        with o4: st.markdown(f"""<div class='metric-card'><div class='metric-label'>Common Defect</div><div class='metric-value'>{stats['most_frequent_defect']}</div></div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualization Grid
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("<div class='custom-subheader'>Process Stability</div>", unsafe_allow_html=True)
            st.pyplot(analytics.get_status_distribution_chart(df))
        with v2:
            st.markdown("<div class='custom-subheader'>Root Cause Analysis</div>", unsafe_allow_html=True)
            st.pyplot(analytics.get_defect_distribution_chart(df))
            
        st.markdown("<div class='custom-subheader'>Process Trends (Anomaly Score)</div>", unsafe_allow_html=True)
        st.pyplot(analytics.get_trend_chart(df))
        
        # PDF Generation
        st.markdown("---")
        st.markdown("<div class='custom-subheader'>Compliance Documentation</div>", unsafe_allow_html=True)
        if st.button("📄 Generate PDF Audit Report"):
            charts = [analytics.get_status_distribution_chart(df), analytics.get_defect_distribution_chart(df), analytics.get_trend_chart(df)]
            report_file = report.generate_qa_report(period, stats, df[df['status'] == 'Defect'], charts)
            with open(report_file, "rb") as f:
                st.download_button("📥 Export Audit PDF", f, file_name=report_file)

# ==========================================
# PAGE 3: SYSTEM INFO
# ==========================================
elif page == "📝 System Info":
    st.markdown("<h1 class='dashboard-title'>System Configuration</h1>", unsafe_allow_html=True)
    
    col_info, col_img = st.columns([1, 1])
    with col_info:
        st.markdown("""
        ### Integrated QA Architecture
        This system combines state-of-the-art anomaly detection with fine-grained classification.
        
        - **Primary Model**: PatchCore (WideResNet-50 backbone)
        - **Classifier**: ResNet-18 (Transfer Learning)
        - **Database**: SQLite3 Secure Logging
        - **Backend**: Python 3.12 / PyTorch 2.x
        
        ### Operational Thresholds
        - **Anomaly Threshold**: 45.0 (Calibrated for MVTecAD Capsule)
        - **Minor Severity**: < 60
        - **Medium Severity**: 60 - 80
        - **Critical Severity**: > 80
        """)
    
    with col_img:
        st.image("https://images.unsplash.com/photo-1587854692152-cbe660dbbb88?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", caption="Industrial QA Environment")