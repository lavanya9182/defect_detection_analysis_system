# Capsule Defect Detection

This project works on the MVTec Capsule dataset using the PatchCore model for anomaly detection.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Dataset**:
    Ensure the dataset is located at `dataset/capsule`.

## Training

To train the model (this performs feature extraction and coreset sampling):

```bash
python train.py
```

Results will be saved in `results/`.
The trained model weights for inference will be at `results/weights/torch/model.pt`.

## Running the App

To start the Streamlit interface:

```bash
streamlit run app.py
```

## Usage
1.  Open the Streamlit app in your browser.
2.  Upload an image of a capsule.
3.  Click "Analyze Image".
4.  View the anomaly score, classification, and segmentation mask.
