import json
import matplotlib.pyplot as plt
from pathlib import Path

history_path = Path("artifacts/models/training_history.json")
output_dir = Path("report_figures")
output_dir.mkdir(parents=True, exist_ok=True)

with open(history_path, "r") as file:
    history = json.load(file)

epochs = range(1, len(history["train_accuracy"]) + 1)

# Accuracy graph
plt.figure(figsize=(9, 5))
plt.plot(epochs, history["train_accuracy"], marker="o",
         label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], marker="s",
         label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.xticks(list(epochs))
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "figure2a_accuracy.png",
            dpi=300, bbox_inches="tight")
plt.close()

# Loss graph
plt.figure(figsize=(9, 5))
plt.plot(epochs, history["train_loss"], marker="o",
         label="Training Loss")
plt.plot(epochs, history["val_loss"], marker="s",
         label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.xticks(list(epochs))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "figure2b_loss.png",
            dpi=300, bbox_inches="tight")
plt.close()

print("Both training graphs generated successfully!")
print("Saved in:", output_dir.resolve())
