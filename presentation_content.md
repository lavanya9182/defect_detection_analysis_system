# PPT Slide Content: Defect Detection & Analysis System

## Topic 1: System Architecture
### Slide 1: Finalized Architecture Diagram
- **Hybrid Core**: Dual-stage pipeline using PatchCore (Anomaly detection) and ResNet-18 (Classification).
- **Process Flow**: Image Acquisition → Feature Extraction → Coreset Comparison → Defect Mask Generation → Final Labeling.
- **Diagram**: Includes specialized stages for high-precision pharmaceutical inspection.

### Slide 2: Database Schema (SQLite3)
- **Table**: `inspections` handles all persistent logging.
- **Key Fields**:
  - `image_id`: Unique session identifier.
  - `status`: Binary classification (Good/Defect).
  - `score`: Quantitative anomaly degree.
  - `severity`: Automated risk assessment (Normal/Minor/Medium/Critical).
  - `timestamp`: Full audit trail tracking.

### Slide 3: Changes from Review-1 Design
- **Architectural Shift**: Transitioned from a standard CNN to a Hybrid Patch-based model for better "One-Class" performance.
- **UI Redesign**: Upgraded to a "Professional Dashboard" with Glassmorphism and sidebar-driven navigation.
- **Feature Addition**: Integrated automated "Certificate of Analysis" PDF generation.

---

## Topic 2: Dataset Details
### Slide 4: Source & Size
- **Source**: MVTec Anomaly Detection (MVTec AD) - Specialized Dataset for Pharmaceutical Quality.
- **Size**: 351 Total high-resolution capsule images.
- **Distribution**: 219 instances for unsupervised training; 132 for validation across 5 different defect classes.

### Slide 5: Features & Preprocessing
- **Image Features**: 256x256 RGB format capturing hairline cracks and surface punctures.
- **Preprocessing Steps**:
  - Center-cropping and dynamic resizing.
  - Pixel-level normalization ($0.485$ mean / $0.229$ std).
  - Data augmentation (rotation/flips) solely for the secondary classifier.
- **Tools Used**: PyTorch (Torchvision), OpenCV (Image processing), Anomalib (Anomalous Feature Modeling).

---

## Topic 3: Algorithms / Models Implemented
### Slide 6: PatchCore (Primary Anomaly Detector)
- **Concept**: Uses a pre-trained WideResNet-50 backbone to map "typical" feature distributions.
- **Coreset Sampling**: Stores only the most descriptive normal features to ensure high speed and low memory usage.
- **Logic**: If a new sample's features are "far" from the coreset, it is flagged as a defect.

### Slide 7: ResNet-18 (Secondary Classifier)
- **Concept**: Activated only when an anomaly is detected.
- **Role**: Fine-tuned classification to distinguish between "Crack", "Poke", and "Squeeze".
- **Benefit**: Provides actionable root-cause feedback to production managers.

### Slide 8: System Flowcharts
- **Step 1**: Image Input Stream.
- **Step 2**: Global Anomaly Score Computation.
- **Step 3**: Binary Decision Gate.
- **Step 4**: Defect Type Recognition & DB Persistence.

---

## Topic 4: Implementation Status
### Slide 9: Modules Completed (With Proof)
- **Inference Module**: PatchCore integration successful (verified with `test_inference.py`).
- **Analytics Module**: Real-time chart generation and data syncing functional.
- **Reporting Module**: Programmatic PDF export with localized charts (`report.py`).
- **Persistence Layer**: SQLite3 database with asynchronous logging.

### Slide 10: UI Demonstration & Structure
- **UX**: Pro-grade Glassmorphism theme with cinematic lighting and Outfit typography.
- **Structure**:
  - `app.py`: Reactive dashboard entry.
  - `db.py`: Secure data layer.
  - `analytics.py`: Visualization engine.
  - `results/`: Trained weights storage.

---

## Topic 5: Testing Strategy
### Slide 11: Unit & Integration Testing
- **Unit Isolation**: Verified individual model loading and DB write performance in controlled environments.
- **Integration**: Tested the recursive update cycle between the Inference screen and the Analytics dashboard.

### Slide 12: Test Cases & Sample Results
- **Test Case 01**: Good capsule validation (Success: Score < 30).
- **Test Case 15**: Critical Crack detection (Success: Score > 80 + Correct Label).
- **Result Proof**: Inspections are logged with 100% fidelity into `capsule_inspections.db`.

---

## Topic 6: Preliminary Results
### Slide 13: Screenshots & Visual Proof
- **Localization**: System successfully generates "Heatmaps" that pinpoint the exact coordinate of defects.
- **Metrics**:
  - **Sensitivity**: 98%+ on MVTec benchmarks.
  - **Latency**: Under 500ms per scan for real-time throughput.
- **Accuracy**: Robust categorization of 5 distinct capsule defect types.

---

## Topic 7: Challenges & Future Scope
### Slide 14: Technical Challenges
- **Weight Reconstruction**: Managing large model binaries (split into parts for cloud compatibility).
- **False Positives**: Calibrating threshold filters to distinguish between surface dust and actual cracks.

### Slide 15: Future Enhancements
- **Hardware Triggering**: Interfacing with pneumatic air jets for physical defect rejection.
- **Edge Integration**: Porting to ARM-based devices (Jetson Nano) for on-site factory deployment.
- **Centralized Cloud**: Multi-factory monitoring via AWS/GCP data sync.
