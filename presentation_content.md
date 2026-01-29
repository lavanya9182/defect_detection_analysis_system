# Detailed Presentation Content

## Slide 1: Title Slide
**Title:** Deep Learning-Based Defect Detection & Analytics for Pharmaceutical Capsules
**Subtitle:** A Hybrid Approach using Unsupervised Anomaly Detection (PatchCore) and Supervised Classification
**Team Details:**
*   **Team Members:** [Insert Names]
*   **Guide:** [Insert Guide Name]
*   **Department:** Computer Science & Engineering / [Your Dept]

---

## Slide 2: Introduction
**Context:**
*   **Pharmaceutical Quality Assurance (QA):** Critical requirement for "Zero-Defect" tolerance in capsule production.
*   **Regulatory Compliance:** Adherence to FDA/GMP standards for visual inspection.
**Current Paradigm:**
*   **Automated Visual Inspection (AVI):** Transition from manual to computer vision-based systems.
*   **Challenge:** High variability in defect types (cracks, dents, discolorations) and rare occurrence of anomalies makes traditional supervised learning inefficient.
**Proposed Solution:**
*   **Hybrid Deep Learning Framework:** Integrating Unsupervised Anomaly Detection (for effective outlier detection) with Supervised Classification (for root cause analysis).

---

## Slide 3: Problem Statement
**Limitations of Manual Inspection:**
*   **Human Error:** Fatigue-induced variability leads to inconsistent defect detection rates (approx. 80-85% reliability).
*   **Throughput Constraint:** Manual inspectors cannot match high-speed production lines (up to 100,000 capsules/hour).
**Limitations of Existing Automation:**
*   **Rule-Based CV:** Sensitive to lighting variations and object rotation; requires rigid feature engineering.
*   **Supervised DL:** Requires massive, balanced datasets for *every* potential defect type, which is impractical in production environments where defects are rare anomalies.

---

## Slide 4: Motivation
**Critical Drivers:**
1.  **Patient Safety:** Even minor physical defects (e.g., holes) can accelerate drug degradation or alter bioavailability.
2.  **Cost of Quality (CoQ):**
    *   **False Negatives (Type II Error):** Releasing bad batches leads to recalls and reputational damage.
    *   **False Positives (Type I Error):** High scrap rates waste expensive raw materials.
3.  **Technological Necessity:** Need for a system capable of "One-Class Classification"—learning only from "Good" samples to detect *any* deviation, including previously unseen defects.

---

## Slide 5: Project Objectives
1.  **Develop an AVI System:** Implement a Deep Learning pipeline for real-time capsule inspection.
2.  **Unsupervised Anomaly Detection:** Utilize **PatchCore** (State-of-the-Art) to identify anomalies without training on defective data.
3.  **Defect Categorization:** Integrate a **ResNet-18** classifier to label detected anomalies (e.g., "Crack", "Poke", "Missing Imprint") for process feedback.
4.  **Analytics & Reporting:** Build a **Streamlit** dashboard with **SQLite** backend for real-time monitoring, trend analysis, and automated **PDF QA Reporting**.

---

## Slide 6: Literature Survey - Evolution of Techniques
*   **Gen 1: Traditional Computer Vision:**
    *   Techniques: Canny Edge Detection, Morphological Operations, Template Matching.
    *   Drawback: Lack of generalization; fails with slight rotations or lighting changes.
*   **Gen 2: Supervised Deep Learning:**
    *   Models: CNNs (YOLO, MobileNet, VGG).
    *   Drawback: "Cold Start" problem—cannot detect defects not present in the training set.
*   **Gen 3: Unsupervised / Semi-Supervised:**
    *   Approaches: Autoencoders (Reconstruction Loss), GANs (AnoGAN).
    *   Drawback: "Blurry" reconstructions can mask subtle high-frequency defects like hairline cracks.

---

## Slide 7: Literature Survey - Gap Analysis & Solution
**Why PatchCore?**
*   **Architecture:** Uses a memory bank of locally aware features extracted from a pre-trained backnone (WideResNet-50).
*   **Advantage:**
    *   **Coreset Sampling:** Reduces memory bank size while preserving feature diversity.
    *   **Nearest Neighbor Search:** fast inference suitable for real-time applications.
    *   **Performance:** Achieves SOTA (99.1% AUC) on the MVTec AD benchmark.
*   **Implementation:** Adopted via the **Anomalib** library (Intel) for standardized training and deployment.

---

## Slide 8: Proposed Methodology - Hybrid Overview
**The 2-Stage Pipeline:**
1.  **Stage 1: Anomaly Detection (Screening):**
    *   **Input:** Raw Capsule Image (256x256).
    *   **Action:** Binary Classification (Good vs. Defect) based on anomaly score.
    *   **Output:** Anomaly Heatmap + Score.
2.  **Stage 2: Defect Classification (Diagnosis):**
    *   **Trigger:** Only activated if Stage 1 Score > Threshold.
    *   **Action:** Multi-class classification via Transfer Learning.
    *   **Output:** Specific Defect Class (e.g., Crack, Poke).
3.  **Stage 3: Analytics Integration:**
    *   Data logging, visualization, and report generation.

---

## Slide 9: System Architecture
*(Refer to Block Diagram)*
**Data Flow Logic:**
*   **Image Acquisition Layer:** Simulated input stream of capsule images.
*   **Processing Layer:**
    *   **Backbone:** `WideResNet-50` acts as the feature extractor.
    *   **Memory Bank:** Stores "Golden Standard" feature configurations.
*   **Decision Layer:**
    *   Computed Euclidean distance to nearest typical neighbor determines the **Anomaly Score**.
    *   Global Max Pooling consolidates pixel-scores into image-scores.
*   **Application Layer:** Streamlit UI for operator interaction and data visualization.

---

## Slide 10: Stage 1 - The Anomaly Detection Core
**Technical Specifications:**
*   **Model:** PatchCore (Patch-based Inspection via Coreset Sampling).
*   **Backbone:** WideResNet-50 (Pre-trained on ImageNet).
*   **Feature Extraction:** Extracts features from intermediate layers (Layer 2 and 3) to capture both low-level textures and high-level semantics.
*   **Mechanism:**
    1.  Divide image into patches.
    2.  Map patches to the Memory Bank.
    3.  Compute local anomaly score based on distance to nearest non-defective patch.
*   **Localization:** Upsamples the patch-wise scores to generate a pixel-perfect **Anomaly Heatmap**.

---

## Slide 11: Stage 2 - Automatic Defect Classification
**Purpose:** Root Cause Analysis (RCA). Knowing *that* a capsule is bad is safety; knowing *why* is process improvement.
**Model:** **ResNet-18** (Residual Neural Network).
**Training Strategy:** Transfer Learning.
*   Pre-trained weights frozen.
*   Fully Connected (FC) head replaced to output N classes.
**Classes:**
*   `Crack`, `Poke` (Puncture), `Squeeze`, `Faulty Imprint`, `Scratch`.
**Optimization:** This stage is skipped for "Good" samples, optimizing computational resources.

---

## Slide 12: Software Implementation Stack
**Development Environment:**
*   **Language:** Python 3.12
*   **Deep Learning Framework:** PyTorch 2.x
*   **Anomaly Library:** Anomalib (Intel OpenVINO ecosystem)
**Application & Analytics:**
*   **Backend Database:** SQLite3 (Lightweight, serverless relational DB).
*   **Frontend/Dashboard:** Streamlit (Reactive web framework).
*   **Data Processing:** Pandas (Time-series analytics), NumPy.
*   **Reporting Engine:** FPDF (Programmatic PDF generation), Matplotlib (Auto-plotting).

---

## Slide 13: Results & Analytics Dashboard
**Key Deliverables:**
1.  **Heatmap Visualization:**
    *   Visual proof of defect localization (Red zones indicate high anomaly scores).
    *   Allows operators to verify "where" the defect is.
2.  **Operational Metrics:**
    *   **Defect Rate:** Real-time calculation ($\frac{Defects}{Total} \times 100$).
    *   **Pareto Analysis:** Histogram of Defect Types (identifying most frequent failures).
3.  **Automated Reporting:**
    *   One-click generation of PDF reports containing production summaries, Gantt charts, and quality graphs.

---

## Slide 14: Work Plan / Timeline
**(Refer to Gantt Chart)**
*   **Phases Completed:**
    *   Phase 1: Literature Survey & Dataset Preparation (MVTec AD / Capsule Dataset).
    *   Phase 2: Model Training (PatchCore) & Validation.
    *   Phase 3: Development of Analytics Module & Dashboard.
*   **Current Status:** Integration & System Testing.
*   **Upcoming (Phase 5):** Final Optimization & Documentation/Thesis writing.

---

## Slide 15: Conclusion & Future Scope
**Conclusion:**
*   Successfully implemented a 'Zero-Shot' anomaly detection system adequate for pharma standards.
*   Eliminated the need for massive defective datasets while maintaining high sensitivity.
**Future Scope:**
*   **Edge Deployment:** Porting model to **NVIDIA Jetson Nano** using **TensorRT** optimization for edge inference.
*   **Hardware Integration:** Interfacing with PLC (Programmable Logic Controller) for physical rejection mechanisms (air jets).
*   **Active Learning:** User feedback loop to retrain the classifier on new defect types.
