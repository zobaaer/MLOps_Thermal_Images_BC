import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
model_path = r"C:\Users\ahmd\Documents\MLOps_Thermal_Images_BC\Streamlit Docker MLOps\model\classif_model.pth"
test_dir = r"C:\Users\ahmd\Documents\MLOps_Thermal_Images_BC\Classification\test"

# Image transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load test dataset
# Assumes test_dir has subfolders 'healthy' and 'sick'
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes
print(f"Test classes: {class_names}")

# Load model
num_classes = len(class_names)
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Evaluation
correct = 0
correct_per_class = [0 for _ in range(num_classes)]
total_per_class = [0 for _ in range(num_classes)]
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(labels.size(0)):
            label = labels[i].item()
            total_per_class[label] += 1
            if predicted[i] == label:
                correct_per_class[label] += 1

# Print results
overall_acc = 100 * correct / total if total > 0 else 0
print(f"Overall accuracy: {overall_acc:.2f}% ({correct}/{total})")
for idx, class_name in enumerate(class_names):
    acc = 100 * correct_per_class[idx] / total_per_class[idx] if total_per_class[idx] > 0 else 0
    print(f"Accuracy for class '{class_name}': {acc:.2f}% ({correct_per_class[idx]}/{total_per_class[idx]})")
