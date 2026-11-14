

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Create output directory
output_dir = Path("report_figures")
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis("off")

ax.text(
    8, 6.5,
    "EfficientNet-B0 Architecture for Indian Currency Classification",
    ha="center", fontsize=16, fontweight="bold"
)

blocks = [
    (0.3, 3.1, 1.7, 1.3, "Input Image",
     "224 × 224 × 3\nRGB"),
    (2.3, 3.1, 1.7, 1.3, "Stem",
     "Initial convolution\nFeature extraction"),
    (4.3, 3.1, 3.0, 1.3, "MBConv Blocks",
     "Mobile inverted bottleneck\nSqueeze-and-Excitation\nFeature extraction"),
    (7.6, 3.1, 1.8, 1.3, "Convolution Head",
     "1 × 1 convolution\nFeature mapping"),
    (9.8, 3.1, 1.8, 1.3, "Global Average\nPooling",
     "Feature reduction"),
    (11.9, 3.1, 1.6, 1.3, "Classifier",
     "Dropout\nFully connected"),
    (13.8, 3.1, 1.9, 1.3, "Output",
     "7 Classes")
]

# Draw architecture blocks
for x, y, w, h, title, description in blocks:

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        linewidth=1.5,
        edgecolor="navy",
        facecolor="lightblue"
    )

    ax.add_patch(box)

    ax.text(
        x + w/2, y + 0.88,
        title,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold"
    )

    ax.text(
        x + w/2, y + 0.38,
        description,
        ha="center",
        va="center",
        fontsize=8
    )

# Draw arrows
for i in range(len(blocks) - 1):

    x1 = blocks[i][0] + blocks[i][2]
    x2 = blocks[i+1][0]

    arrow = FancyArrowPatch(
        (x1 + 0.03, 3.75),
        (x2 - 0.03, 3.75),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.5,
        color="black"
    )

    ax.add_patch(arrow)

# Output classes
ax.text(
    8, 2.3,
    "Seven Output Classes",
    ha="center",
    fontsize=12,
    fontweight="bold"
)

classes = [
    "₹10", "₹20", "₹50", "₹100",
    "₹200", "₹500", "Background"
]

for i, label in enumerate(classes):

    x = 2.1 + i * 1.7

    box = FancyBboxPatch(
        (x, 1.2), 1.45, 0.65,
        boxstyle="round,pad=0.05",
        edgecolor="darkgreen",
        facecolor="lightgreen"
    )

    ax.add_patch(box)

    ax.text(
        x + 0.725, 1.525,
        label,
        ha="center",
        va="center",
        fontsize=9
    )

ax.text(
    8, 0.5,
    "EfficientNet-B0 CNN-based image classification model",
    ha="center",
    fontsize=10,
    style="italic"
)

plt.tight_layout()

plt.savefig(
    output_dir / "figure3_efficientnet_architecture.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure 3 generated successfully!")
print("Saved at: report_figures/figure3_efficientnet_architecture.png")
