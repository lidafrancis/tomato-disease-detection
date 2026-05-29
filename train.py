# train.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset_loader import train_loader, val_loader, CLASS_NAMES
from model import build_model

EPOCHS    = 10
LR        = 0.001
PATIENCE  = 5
SAVE_PATH = "best_model.pth"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 50)
print(f"  Device        : {DEVICE}")
print(f"  Epochs        : {EPOCHS}")
print(f"  Learning rate : {LR}")
print(f"  Batch size    : 64")
print(f"  Image size    : 224x224")
print(f"  Classes       : {len(CLASS_NAMES)} — {CLASS_NAMES}")
print("=" * 50)

model = build_model(num_classes=len(CLASS_NAMES)).to(DEVICE)

criterion = nn.CrossEntropyLoss()

# Only train the final fc layer — clean and simple
optimizer = torch.optim.Adam(
    model.fc.parameters(), lr=LR, weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)


def train_one_epoch():
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"    batch {batch_idx+1:>3}/{len(train_loader)}"
                  f" | loss: {loss.item():.4f}")

    return total_loss / len(train_loader), correct / total * 100


def validate():
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images  = images.to(DEVICE)
            labels  = labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(val_loader), correct / total * 100


train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_loss = float('inf')
best_epoch    = 0
no_improve    = 0

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("-" * 40)

    t_loss, t_acc = train_one_epoch()
    v_loss, v_acc = validate()

    train_losses.append(t_loss)
    val_losses.append(v_loss)
    train_accs.append(t_acc)
    val_accs.append(v_acc)

    scheduler.step(v_loss)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"\n  Train — loss: {t_loss:.4f}  acc: {t_acc:.1f}%")
    print(f"  Val   — loss: {v_loss:.4f}  acc: {v_acc:.1f}%")
    print(f"  LR    : {current_lr:.6f}")

    if v_loss < best_val_loss:
        best_val_loss = v_loss
        best_epoch    = epoch
        no_improve    = 0
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss':             v_loss,
            'val_acc':              v_acc,
            'class_names':          CLASS_NAMES,
        }, SAVE_PATH)
        print(f"  ✓ Best model saved — val_loss: {v_loss:.4f}")
    else:
        no_improve += 1
        print(f"  No improvement ({no_improve}/{PATIENCE})")

    if no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

epochs_ran = range(1, len(train_losses) + 1)
fig, axes  = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_ran, train_losses, label="Train loss",
             color="#2563eb", linewidth=2, marker='o', markersize=6)
axes[0].plot(epochs_ran, val_losses, label="Val loss",
             color="#dc2626", linewidth=2, marker='s', markersize=6)
axes[0].axvline(x=best_epoch, color="#16a34a", linestyle="--",
                linewidth=1.5, label=f"Best epoch ({best_epoch})")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curve", fontweight='bold')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_ran, train_accs, label="Train acc",
             color="#2563eb", linewidth=2, marker='o', markersize=6)
axes[1].plot(epochs_ran, val_accs, label="Val acc",
             color="#dc2626", linewidth=2, marker='s', markersize=6)
axes[1].axvline(x=best_epoch, color="#16a34a", linestyle="--",
                linewidth=1.5, label=f"Best epoch ({best_epoch})")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Accuracy Curve", fontweight='bold')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 100)

plt.suptitle("Tomato 3-Class ResNet18 — Training Results",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*50}")
print(f"  Training Complete!")
print(f"  Best epoch   : {best_epoch}")
print(f"  Best val acc : {val_accs[best_epoch-1]:.1f}%")
print(f"  Best val loss: {best_val_loss:.4f}")
print(f"  Model saved  : {SAVE_PATH}")
print(f"  Graph saved  : loss_curve.png")
print(f"{'='*50}")