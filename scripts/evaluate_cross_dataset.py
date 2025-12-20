import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATA_DIR = "data/processed/faces"
MODEL_PATH = "models/baseline_efficientnet.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")
