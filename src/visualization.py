import json
import matplotlib.pyplot as plt


with open("../training_graphs/history.json", "r") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

# ===========================
# Loss Curve
# ===========================
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    marker="o",
    label="Train Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig("../training_graphs/loss_curve.png", dpi=300)

plt.show()

# ===========================
# Accuracy Curve
# ===========================
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig("../training_graphs/accuracy_curve.png", dpi=300)

plt.show()

# ===========================
# F1 Curve
# ===========================
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["f1"],
    marker="o",
    label="Validation F1 Score"
)

plt.title("Validation F1 Score")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig("../training_graphs/f1_curve.png", dpi=300)

plt.show()

print("Graphs saved successfully!")