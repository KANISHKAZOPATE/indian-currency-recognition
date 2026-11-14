import matplotlib.pyplot as plt
import numpy as np

classes = ["?10", "?100", "?20", "?200", "?50", "?500", "Background"]

cm = np.array([
    [7, 0, 0, 0, 0, 0, 0],
    [0, 14, 0, 0, 0, 0, 0],
    [0, 0, 11, 1, 0, 0, 0],
    [0, 0, 0, 15, 0, 0, 0],
    [0, 0, 0, 0, 13, 0, 0],
    [0, 0, 0, 0, 0, 11, 0],
    [0, 0, 0, 0, 0, 0, 8]
])

fig, ax = plt.subplots(figsize=(9, 7))
image = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=35, ha="right")
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title("Confusion Matrix - EfficientNet-B0")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > 8 else "black",
                fontsize=11)

fig.colorbar(image, ax=ax, label="Number of Images")
plt.tight_layout()
plt.savefig("report_figures/figure1_confusion_matrix.png",
            dpi=300, bbox_inches="tight")
plt.show()

print("Figure 1 saved successfully!")
