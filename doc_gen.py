from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from datetime import datetime
import tempfile

# Ensure output directory exists
OUTPUT_DIR = "results/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Project Review-1 Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'Defect Detection & Analytics System for Pharmaceutical Capsules', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 8, body)
        self.ln()

    def add_image_centered(self, image_path, w=150):
        if os.path.exists(image_path):
            self.image(image_path, x=(210-w)/2, w=w)
            self.ln(10)

def generate_gantt_chart(filename):
    """Generates a Gantt chart for the Work Plan."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = [
        ("Literature Survey & Data Collection", 0, 2),
        ("Model Selection (PatchCore) & Training", 1, 2),
        ("UI Development (Streamlit)", 3, 2),
        ("Analytics & DB Module", 4, 2),
        ("Auto-Classification (ResNet)", 5, 2),
        ("System Integration & Testing", 7, 2),
        ("Final Documentation & Thesis", 9, 1)
    ]
    
    # Y-axis positions
    y_pos = [i * 10 for i in range(len(tasks))]
    
    for i, (task, start, duration) in enumerate(tasks):
        ax.broken_barh([(start, duration)], (y_pos[i]-4, 8), facecolors='#4c72b0')
        ax.text(start + duration + 0.5, y_pos[i], f"{duration} Weeks", va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[0] for t in tasks])
    ax.set_xlabel('Weeks')
    ax.set_xlim(0, 11) # Set limit to slightly over 10 to show end
    ax.set_title('Project Work Plan (10 Weeks)')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_block_diagram(filename):
    """Generates a publication-quality System Architecture Diagram."""
    # High resolution, wide aspect ratio
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Professional Style Config
    box_style = dict(boxstyle="round,pad=0.4", ec="#333333", lw=1.5)
    
    # Color Palette (Pastel/Professional)
    colors = {
        "input": "#E3F2FD",      # Light Blue
        "process": "#E8F5E9",    # Light Green
        "model": "#FFF3E0",      # Light Orange
        "decision": "#FCE4EC",   # Light Pink
        "output": "#F3E5F5",     # Light Purple
        "db": "#ECEFF1"          # Light Grey
    }

    # Helper: Draw Node
    def draw_node(x, y, w, h, text, type_key, label=None):
        color = colors.get(type_key, "#ffffff")
        # Drop Shadow
        shadow = patches.FancyBboxPatch((x+0.1, y-0.1), w, h, fc='#DDDDDD', ec='none', boxstyle="round,pad=0.4")
        ax.add_patch(shadow)
        # Box
        rect = patches.FancyBboxPatch((x, y), w, h, fc=color, **box_style)
        ax.add_patch(rect)
        # Text
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, weight='bold', wrap=True)
        # Sub-label
        if label:
             ax.text(x, y + h + 0.1, label, ha='left', va='bottom', fontsize=8, style='italic', color='#555555')

    # Helper: Draw Arrow
    def draw_connect(x1, y1, x2, y2, text=None, curve=0.0):
        # connectionstyle=f"arc3,rad={curve}"
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>,head_length=0.4,head_width=0.2",
            connectionstyle=f"arc3,rad={curve}",
            color="#444444", lw=1.5
        )
        ax.add_patch(arrow)
        if text:
            # Simple midpoint calculation for text placement
            mx, my = (x1+x2)/2, (y1+y2)/2
            # Adjust for curve
            if curve != 0: my += 0.5 * (1 if curve > 0 else -1)
            
            bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
            ax.text(mx, my, text, ha='center', va='center', fontsize=9, color='black', bbox=bbox)

    # --- Layout Nodes ---
    
    # 1. Input Layer
    draw_node(0.5, 6.0, 2.0, 1.2, "Capsule Image\n(Camera Input)", "input", "Data Acquisition")

    # 2. Preprocessing
    draw_node(3.5, 6.0, 2.0, 1.2, "Preprocessing\n(Resize/Crop)", "process", "Normalization")

    # 3. Anomaly Detection Engine (The Core)
    # Background Group
    core_bg = patches.FancyBboxPatch((6.0, 4.8), 3.0, 2.8, fc='#FFF8E1', ec='#FFB74D', lw=1, boxstyle="round,pad=0.2", linestyle="--")
    ax.add_patch(core_bg)
    ax.text(6.1, 7.8, "Stage 1: Anomaly Detection", fontsize=9, weight='bold', color='#BF360C')

    draw_node(6.2, 6.0, 2.6, 1.2, "PatchCore Backbone\n(WideResNet50)", "model")
    
    # Memory Bank Representation
    draw_node(7.0, 3.5, 1.0, 1.5, "Memory\nBank", "db") 
    
    # 4. Decision Node
    draw_node(9.5, 6.0, 2.0, 1.2, "Thresholding\n(Distance Score)", "decision", "Decision Logic")

    # 5. Classifier (Stage 2)
    # Background Group
    clf_bg = patches.FancyBboxPatch((9.5, 0.5), 2.0, 3.0, fc='#E1F5FE', ec='#0288D1', lw=1, boxstyle="round,pad=0.2", linestyle="--")
    ax.add_patch(clf_bg)
    ax.text(9.6, 3.7, "Stage 2: Classification", fontsize=9, weight='bold', color='#01579B')

    draw_node(9.5, 2.0, 2.0, 1.2, "Defect Classifier\n(ResNet18)", "model")

    # 6. Database
    draw_node(4.0, 2.0, 2.0, 1.2, "SQL Database\n(Logs & Stats)", "db")

    # 7. Output / Dashboard
    draw_node(0.5, 2.0, 2.0, 1.2, "Dashboard UI\n(Analytics & PDF)", "output", "Human Machine Interface")

    # --- Connections ---

    # Input -> Pre
    draw_connect(2.5, 6.6, 3.5, 6.6)
    
    # Pre -> PatchCore
    draw_connect(5.5, 6.6, 6.2, 6.6)
    
    # PatchCore uses Memory Bank
    draw_connect(7.5, 6.0, 7.5, 5.0, text="Nearest Neighbor", curve=0.0)
    
    # PatchCore -> Decision
    draw_connect(8.8, 6.6, 9.5, 6.6, text="Score")
    
    # Decision -> Classifier (DEFECT path)
    draw_connect(10.5, 6.0, 10.5, 3.2, text="If Defect", curve=0.0)
    
    # Decision -> DB (GOOD path) - Long loop back
    draw_connect(10.5, 7.2, 5.0, 3.2, text="If Good", curve=-0.6)
    
    # Classifier -> DB
    draw_connect(9.5, 2.6, 6.0, 2.6, text="Defect Type")

    # DB -> Dashboard
    draw_connect(4.0, 2.6, 2.5, 2.6, text="Fetch Data")
    
    # Dashboard -> Report
    # (Just text annotation)
    
    plt.title("Proposed Integrated Anomaly Detection Architecture", fontsize=16, weight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(filename, dpi=300) # High DPI for paper
    plt.close()

def generate_report():
    pdf = PDF()
    
    # --- 1. Problem Statement & Objectives ---
    pdf.add_page()
    pdf.chapter_title("1. Problem Statement and Objectives")
    
    problem_text = (
        "Problem Statement:\n"
        "Manual inspection of pharmaceutical capsules is a critical quality control process but suffers from significant limitations. "
        "Human inspectors are prone to fatigue, leading to inconsistent defect detection rates. Furthermore, high-speed production lines "
        "exceed human visual processing capabilities, resulting in potential release of defective batches or high scrap rates due to false rejections.\n\n"
        "Motivation:\n"
        "Ensuring 100% quality in pharmaceutical products is mandatory for patient safety. Automated Visual Inspection (AVI) systems offer a reliable, "
        "non-contact, and high-speed alternative. However, traditional AVI systems struggle with subtle defects and require extensive feature engineering.\n\n"
        "Objectives:\n"
        "1. Develop an Automated Visual Inspection system for pharmaceutical capsules using Deep Learning.\n"
        "2. Implement an Unsupervised Anomaly Detection model (PatchCore) to detect unknown defects with high accuracy.\n"
        "3. Integrate a Supervised Classifier (ResNet18) to categorize specific defect types (e.g., Crack, Poke, Squeeze).\n"
        "4. Create a comprehensive Analytics Dashboard for real-time monitoring and reporting.\n"
    )
    pdf.chapter_body(problem_text)

    # --- 2. Literature Review ---
    pdf.chapter_title("2. Detailed Literature Review")
    
    lit_review_text = (
        "The field of anomaly detection in manufacturing has evolved significantly. Early approaches relied on traditional Computer Vision techniques "
        "such as edge detection and morphological operations (image differencing). These methods are computationally efficient but lack robustness against "
        "lighting variations and complex textures.\n\n"
        "Supervised Deep Learning (e.g., CNNs like YOLO, ResNet) improved performance but requires large, fully labeled datasets of defined defects. "
        "In pharmaceutical manufacturing, defective samples are rare, making supervised training difficult due to class imbalance.\n\n"
        "Current State-of-the-Art methodologies focus on Unsupervised Learning, training only on 'Good' samples. "
        "Autoencoders and GANs reconstruct images and detect anomalies based on reconstruction error, but often produce blurry outputs. "
        "Recent advances like SPADE and PatchCore (Roth et al., 2022) utilize pre-trained feature extractors (e.g., WideResNet) and memory banks "
        "of nominal features. PatchCore achieves SOTA performance on the MVTec AD dataset by using coreset sampling to reduce memory footprint while "
        "maintaining high detection accuracy. This project adopts PatchCore for its superior performance and low inference latency."
    )
    pdf.chapter_body(lit_review_text)

    # --- 3. Methodology ---
    pdf.add_page()
    pdf.chapter_title("3. Proposed Methodology / System Design")
    
    method_text = (
        "Our approach integrates Unsupervised Anomaly Detection with Supervised Classification to provide a robust inspection solution.\n\n"
        "System Architecture:\n"
        "1. Image Acquisition: High-resolution images of capsules are captured (simulated via upload).\n"
        "2. Preprocessing: Images are resized to 256x256 and normalized.\n"
        "3. Anomaly Detection (PatchCore): \n"
        "   - Features are extracted using a pre-trained Wide ResNet-50 backbone.\n"
        "   - Local features are compared against a memory bank of 'Good' features using Nearest Neighbor Search.\n"
        "   - An Anomaly Score is generated. If Score > Threshold, it is flagged as specific Defect.\n"
        "4. Defect Classification (ResNet-18): \n"
        "   - Defective images are passed to a secondary classifier trained on specific defect types (Crack, Poke, etc.) to identify the root cause.\n"
        "5. Analytics & Visualization:\n"
        "   - Results are logged to an SQLite database.\n"
        "   - A Streamlit Dashboard visualizes trends, defect rates, and generates PDF reports.\n"
    )
    pdf.chapter_body(method_text)
    
    # Generate and add Block Diagram
    diagram_path = "results/system_diagram.png"
    generate_block_diagram(diagram_path)
    pdf.add_image_centered(diagram_path, w=170)

    # --- 4. Work Plan ---
    pdf.add_page()
    pdf.chapter_title("4. Work Plan / Timeline")
    
    plan_text = (
        "The project is structured into 5 phases over a 10-week period. We have currently completed Phase 3 (Analytics Module) "
        "and are in the Integration & Testing phase."
    )
    pdf.chapter_body(plan_text)
    
    # Generate and add Gantt Chart
    gantt_path = "results/gantt_chart.png"
    generate_gantt_chart(gantt_path)
    pdf.add_image_centered(gantt_path, w=170)
    
    # Save Report
    output_path = os.path.join(OUTPUT_DIR, "Project_Review_1_Report.pdf")
    pdf.output(output_path)
    print(f"Report generated successfully: {output_path}")

    # Cleanup temp images
    if os.path.exists(diagram_path): os.remove(diagram_path)
    if os.path.exists(gantt_path): os.remove(gantt_path)

if __name__ == "__main__":
    generate_report()
