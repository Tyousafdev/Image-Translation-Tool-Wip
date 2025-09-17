# train_bubble_classifier.py
import torch, os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Data ---
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ColorJitter(contrast=0.2, brightness=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_ds = datasets.ImageFolder("dataset/train", transform=transform)
val_ds   = datasets.ImageFolder("dataset/val", transform=transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=32)

# --- Model ---
model = models.resnet18(weights=None, num_classes=2)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Training ---
for epoch in range(1, 31):  # 30 epochs
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    train_acc = correct / total * 100

    # Validation
    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            v_correct += (pred == y).sum().item()
            v_total += y.size(0)
    val_acc = v_correct / v_total * 100

    print(f"Epoch {epoch}/30 | Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

torch.save(model.state_dict(), "bubble_classifier.pth")
print("✅ Saved model to bubble_classifier.pth")
