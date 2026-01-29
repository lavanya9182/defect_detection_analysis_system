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

st.set_page_config(page_title="Capsule Defect Detection", layout="wide")

st.title("💊 Capsule Defect Detection System")

# --- Tabs ---
tab_inference, tab_analytics = st.tabs(["🕵️ Inference & Inspection", "📊 Analytics & Reporting"])

# --- Model Loading ---
@st.cache_resource
def load_model(weights_path):
    if not os.path.exists(weights_path):
        st.error(f"Model weights not found at {weights_path}. Please run training first.")
        return None
    
    inferencer = TorchInferencer(
        path=weights_path,
        device="cpu", 
    )
    return inferencer

# Path where train.py exports the model
MODEL_PATH = "results/weights/weights/torch/model.pt"
model = load_model(MODEL_PATH)

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

        # Defect Type Manual Input (Optional)
        st.markdown("#### Inspection Details")
        defect_type_input = st.selectbox(
            "Select Defect Type (if visible)", 
            ["Normal", "Crack", "Hole", "Color Mismatch", "Contamination", "Deformed", "Other"],
            index=0
        )

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
                            
                    # Manual Override for specific defect type mapping
                    final_defect_type = "Normal"
                    if is_defect:
                         # If user left it as "Normal" but model says Defect, label as "General Defect"
                         if defect_type_input == "Normal":
                             final_defect_type = "General Defect"
                         else:
                             final_defect_type = defect_type_input
                    
                    # --- Results ---
                    st.success("Analysis Complete!")
                    
                    # Log to DB
                    # Generate a unique short ID for the image/inspection
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
            # collect charts
            charts = [chart_status, chart_dist, chart_trend]
            # filter defects df
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
