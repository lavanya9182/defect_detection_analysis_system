import streamlit as st
import os
import sys

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Pharma QA System", layout="wide", page_icon="💊")

# --- INITIAL DIAGNOSTICS ---
st.write("### 🧪 System Initialization")
st.caption("Environment: Streamlit Cloud / Python " + sys.version.split()[0])

try:
    with st.status("Verifying Core Dependencies...", expanded=False) as status:
        import numpy as np
        st.write("✅ NumPy loaded")
        from PIL import Image
        st.write("✅ PIL loaded")
        import torch
        st.write(f"✅ PyTorch {torch.__version__} loaded")
        import cv2
        st.write("✅ OpenCV loaded")
        import tempfile
        import uuid
        from anomalib.deploy import TorchInferencer
        st.write("✅ Anomalib loaded")
        status.update(label="Dependencies Verified!", state="complete")

    # Allow loading pickle models
    os.environ["TRUST_REMOTE_CODE"] = "True"

    with st.status("Initializing Database & Modules...", expanded=False) as status:
        import db
        db.init_db()
        st.write("✅ Database initialized")
        import analytics
        st.write("✅ Analytics logic loaded")
        import report
        st.write("✅ Reporting logic loaded")
        status.update(label="Modules Ready!", state="complete")

    # Professional UI Theme & CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        html, body, [class*="st-at"] { font-family: 'Outfit', sans-serif; }
        .stApp { background: radial-gradient(circle at top left, #1a1c2c, #0d0e1a); color: #ffffff; }
        [data-testid="stSidebar"] { background-color: rgba(20, 22, 39, 0.8) !important; backdrop-filter: blur(10px); border-right: 1px solid rgba(255, 255, 255, 0.1); }
        .metric-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 15px; }
        .metric-label { font-size: 14px; color: #a0a0c0; text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-size: 24px; font-weight: 700; margin-top: 5px; }
        .dashboard-title { font-size: 36px; font-weight: 700; background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 30px; }
        .custom-subheader { font-size: 20px; font-weight: 600; color: #e2e8f0; margin-top: 10px; margin-bottom: 10px; border-left: 4px solid #6366f1; padding-left: 15px; }
        .stButton>button { background: linear-gradient(90deg, #4f46e5, #7c3aed); color: white; border-radius: 8px; width: 100%; }
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
        st.caption("v1.5.0 Professional Edition")

    # --- Model Loading Logic ---
    @st.cache_resource
    def load_models(anomaly_weights_path, classifier_weights_path):
        models = {}
        
        # 1. Anomaly Model (PatchCore)
        if not os.path.exists(anomaly_weights_path):
            part_prefix = anomaly_weights_path + ".part_"
            if os.path.exists(part_prefix + "aa"): 
                try:
                    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(anomaly_weights_path)) as tmp_file:
                        reconstructed_temp = tmp_file.name
                        for suffix in ['aa', 'ab', 'ac']: 
                            part_file = part_prefix + suffix
                            if os.path.exists(part_file):
                                with open(part_file, 'rb') as src:
                                    tmp_file.write(src.read())
                            else:
                                raise FileNotFoundError(f"Missing model part: {part_file}")
                    os.rename(reconstructed_temp, anomaly_weights_path)
                except Exception as e:
                    if 'reconstructed_temp' in locals() and os.path.exists(reconstructed_temp): os.remove(reconstructed_temp)
                    st.error(f"Failed to reconstruct model: {e}")

        if os.path.exists(anomaly_weights_path):
            try:
                models['anomaly'] = TorchInferencer(path=anomaly_weights_path, device="cpu")
            except Exception as e:
                 st.error(f"Error loading Anomaly Model: {e}")
            
        if os.path.exists(classifier_weights_path):
            from torchvision import models as tv_models
            from torchvision import transforms
            import torch.nn as nn
            try:
                checkpoint = torch.load(classifier_weights_path, map_location=torch.device('cpu'))
                class_names = checkpoint['class_names']
                classifier = tv_models.resnet18(weights=None)
                num_ftrs = classifier.fc.in_features
                classifier.fc = nn.Linear(num_ftrs, len(class_names))
                classifier.load_state_dict(checkpoint['model_state_dict'])
                classifier.eval()
                models['classifier'] = classifier
                models['classes'] = class_names
                models['transforms'] = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            except Exception as e:
                st.error(f"Failed to load classifier: {e}")
        return models

    # --- EXECUTION ---
    ANOMALY_MODEL_PATH = "results/weights/weights/torch/model.pt"
    CLASSIFIER_MODEL_PATH = "results/classifier.pth"

    # PAGE 1: INFERENCE
    if page == "🕵️ Real-time Inspection":
        st.markdown("<h1 class='dashboard-title'>Neural Inspection Interface</h1>", unsafe_allow_html=True)
        
        # --- Lazy Load Models ---
        with st.spinner("Initializing Deep Learning Engine..."):
            loaded_models = load_models(ANOMALY_MODEL_PATH, CLASSIFIER_MODEL_PATH)
            model = loaded_models.get('anomaly')
            classifier = loaded_models.get('classifier')
            class_names = loaded_models.get('classes')
            classifier_transforms = loaded_models.get('transforms')
        
        col_input, col_output = st.columns([1, 1.2])
        
        with col_input:
            st.markdown("<div class='custom-subheader'>Image Acquisition</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload Capsule Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                image_np = np.array(image)
                st.image(image, caption="Original Stream", use_column_width=True)
                if st.button("🚀 Analyze Capsule"):
                    if model is None: st.error("Inference module unavailable.")
                    else:
                        with st.spinner("Processing..."):
                            predictions = model.predict(image=image_np)
                            pred_score = predictions.pred_score
                            if isinstance(pred_score, torch.Tensor): pred_score = pred_score.item()
                            anomaly_map = predictions.anomaly_map
                            is_defect = pred_score > 45.0
                            pred_label = "Defect" if is_defect else "Good"
                            
                            severity = "Normal"
                            if is_defect:
                                if pred_score < 60: severity = "Minor"
                                elif pred_score < 80: severity = "Medium"
                                else: severity = "Critical"
                                    
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
                            
                            db.log_inspection(str(uuid.uuid4())[:8], pred_label, final_defect_type, pred_score, severity)
                            
                            with col_output:
                                st.markdown("<div class='custom-subheader'>Inspection Summary</div>", unsafe_allow_html=True)
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

    # PAGE 2: ANALYTICS
    elif page == "📊 Analytics Dashboard":
        st.markdown("<h1 class='dashboard-title'>Product Quality Intelligence</h1>", unsafe_allow_html=True)
        period = st.selectbox("Historical Window", ["All", "Daily", "Weekly", "Monthly"])
        if st.button("🔄 Refresh Data Pipeline"): st.cache_data.clear()
        df = db.fetch_data(period)
        if df.empty: st.info("No data available.")
        else:
            stats = analytics.calculate_stats(df)
            o1, o2, o3, o4 = st.columns(4)
            with o1: st.markdown(f"""<div class='metric-card'><div class='metric-label'>Total Inspected</div><div class='metric-value'>{stats['total']}</div></div>""", unsafe_allow_html=True)
            with o2: st.markdown(f"""<div class='metric-card'><div class='metric-label'>Defect Rate</div><div class='metric-value'>{stats['defect_rate']:.1f}%</div></div>""", unsafe_allow_html=True)
            with o3: st.markdown(f"""<div class='metric-card'><div class='metric-label'>Total Defects</div><div class='metric-value'>{stats['defect_count']}</div></div>""", unsafe_allow_html=True)
            with o4: st.markdown(f"""<div class='metric-card'><div class='metric-label'>Common Defect</div><div class='metric-value'>{stats['most_frequent_defect']}</div></div>""", unsafe_allow_html=True)
            st.markdown("---")
            v1, v2 = st.columns(2)
            with v1: st.pyplot(analytics.get_status_distribution_chart(df))
            with v2: st.pyplot(analytics.get_defect_distribution_chart(df))
            st.pyplot(analytics.get_trend_chart(df))
            
            # --- PDF Generation ---
            st.markdown("---")
            st.markdown("<div class='custom-subheader'>Compliance Documentation</div>", unsafe_allow_html=True)
            if st.button("📄 Generate PDF Audit Report"):
                charts = [analytics.get_status_distribution_chart(df), analytics.get_defect_distribution_chart(df), analytics.get_trend_chart(df)]
                report_file = report.generate_qa_report(period, stats, df[df['status'] == 'Defect'], charts)
                with open(report_file, "rb") as f:
                    st.download_button("📥 Export Audit PDF", f, file_name=report_file)

    # PAGE 3: SYSTEM INFO
    elif page == "📝 System Info":
        st.markdown("<h1 class='dashboard-title'>System Configuration</h1>", unsafe_allow_html=True)
        st.markdown("""
        ### Integrated QA Architecture
        - **Primary Model**: PatchCore (WideResNet-50)
        - **Classifier**: ResNet-18
        - **Environment**: Streamlit Cloud / Ubuntu
        """)

except Exception as e:
    st.error("### 🛑 Initialization Failure")
    st.error(f"Error: {str(e)}")
    import traceback
    st.code(traceback.format_exc())