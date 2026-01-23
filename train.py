import os
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType

def train():
    # 1. Data
    # MVTecAD 
    datamodule = MVTecAD(
        root="./dataset",
        category="capsule",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4
    )

    # 2. Model
    model = Patchcore(
        backbone="wide_resnet50_2",
        pre_trained=True,
    )

    # 3. Engine
    # We use the Engine to train (fit) and test
    engine = Engine(
        default_root_dir="./results",
        accelerator="auto", # auto detect cpu/gpu/mps
        limit_val_batches=0, # Skip validation to avoid known Anomalib bug on some platforms/datasets
        num_sanity_val_steps=0,
    )

    print("Starting training...")
    engine.fit(model=model, datamodule=datamodule)
    
    # Skip testing for now
    # print("Starting testing...")
    # engine.test(model=model, datamodule=datamodule)

    print("Exporting model...")
    # Export to OpenVINO or Torch for inference
    # We will use torch for now for simplicity in Streamlit
    engine.export(
        model=model,
        export_type=ExportType.TORCH,
        export_root="./results/weights",
    )
    
    # Also export as simple .pt if needed, but Engine.export handles it.

if __name__ == "__main__":
    train()
