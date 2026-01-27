import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import os
import shutil

def train_classifier():
    # Configuration
    DATA_DIR = "./dataset/capsule/test"
    MODEL_SAVE_PATH = "./results/classifier.pth"
    NUM_EPOCHS = 10  # Very small dataset, 10-20 epochs usually enough for transfer learning
    BATCH_SIZE = 8
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return

    # Data Transforms (Augmentation is key for small data)
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Dataset
    # We want to exclude 'good' folder usually if we only classify defects, 
    # but ImageFolder loads everything. We can filter later or just ignore the 'good' class output.
    # Actually, for simplicity, let's load everything. 
    # If the user uploads a 'Good' image and PatchCore fails (says Defect), the classifier saying "Good" is a useful recovery.
    # So we will train on ALL classes including 'good'.
    
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    
    dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Model Setup (ResNet18)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print(f"Using device: {device}")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Replace last layer
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training Loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        corrects = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(full_dataset)
        epoch_acc = corrects.float() / len(full_dataset)

        print(f'Epoch {epoch}/{NUM_EPOCHS - 1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print("Training complete.")
    
    # Save Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Save model state dict AND class names
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_classifier()
