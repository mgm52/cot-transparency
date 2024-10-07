import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Colors for the bars
colors = ["#FFDDA6", "#FE7070", "#CFE4FF"]  # Reversed order to match models

# Datasets
datasets = ["Hellaswag", "LogiQA", "MMLU", "TruthfulQA"]

# Data for each dataset and model
# 2400 samples!
gpt_4o_mini_biased = [13.71, 51.44, 19.83, 26.35]
gpt_4o_mini_unbiased = [13.45, 50.59, 19.77, 26.06]

control_biased = [18.06, 54.36, 21.51, 25.76]
control_unbiased = [13.53, 50.84, 21.85, 24.07]

bct_biased = [15.44, 54.48, 21.87, 27.01]
bct_unbiased = [15.55, 54.31, 21.48, 25.84]

# Number of datasets
n_datasets = len(datasets)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 7))  # Increased figure size for better readability

# Define bar width and positions for each group
bar_width = 0.2
indices = np.arange(n_datasets)

# Plot bars for BCT
ax.bar(indices - bar_width, bct_biased, bar_width, color=colors[0], edgecolor="black", label="BCT Biased")

# Plot bars for Control
ax.bar(indices, control_biased, bar_width, color=colors[1], edgecolor="black", label="Control Biased")

# Plot bars for GPT-4o-mini
ax.bar(
    indices + bar_width, gpt_4o_mini_biased, bar_width, color=colors[2], edgecolor="black", label="GPT-4o-mini Biased"
)

# Plot horizontal dashed lines for GPT-4o-mini unbiased averages
dashed_line_width = bar_width * 0.8  # Dashed line spans 80% of the bar width
for i, unbiased_value in enumerate(gpt_4o_mini_unbiased):
    x_center = indices[i] + bar_width
    x_start = x_center - dashed_line_width / 2
    x_end = x_center + dashed_line_width / 2
    ax.plot(
        [x_start, x_end],
        [unbiased_value, unbiased_value],
        color="black",
        linestyle="dashed",
        linewidth=2,
        label="GPT-4o-mini Unbiased" if i == 0 else "",
    )

# Plot horizontal dashed lines for Control unbiased averages
for i, unbiased_value in enumerate(control_unbiased):
    x_center = indices[i]  # Control bars are centered at `indices`
    x_start = x_center - dashed_line_width / 2
    x_end = x_center + dashed_line_width / 2
    ax.plot(
        [x_start, x_end],
        [unbiased_value, unbiased_value],
        color="black",
        linestyle="dashed",
        linewidth=2,
        label="Control Unbiased" if i == 0 else "",
    )

# Plot horizontal dashed lines for BCT unbiased averages
for i, unbiased_value in enumerate(bct_unbiased):
    x_center = indices[i] - bar_width  # BCT bars are at `indices - bar_width`
    x_start = x_center - dashed_line_width / 2
    x_end = x_center + dashed_line_width / 2
    ax.plot(
        [x_start, x_end],
        [unbiased_value, unbiased_value],
        color="black",
        linestyle="dashed",
        linewidth=2,
        label="BCT Unbiased" if i == 0 else "",
    )

# Add labels and title
ax.set_xlabel("Datasets", fontsize=12)
ax.set_ylabel("Incorrect %", fontsize=12)
ax.set_title("Sarcasm-Smart Bias Across Different Datasets", fontsize=14)
ax.set_xticks(indices)
ax.set_xticklabels(datasets, fontsize=12)

# Define y-axis limits
ax.set_ylim(0, 60)

# Update the legend to reflect the new order
custom_legend = [
    Line2D([0], [0], color=colors[0], lw=4, label="BCT"),
    Line2D([0], [0], color=colors[1], lw=4, label="Control"),
    Line2D([0], [0], color=colors[2], lw=4, label="GPT-4o-mini"),
    Line2D([0], [0], color="black", linestyle="dashed", lw=2, label="BCT/Control/GPT-4o-mini Unbiased"),
]

ax.legend(handles=custom_legend, fontsize=12)

# Add grid lines for better readability
ax.grid(True, which="both", linestyle="--", linewidth=0.5, axis="y", alpha=0.7)

# Enhance layout
plt.tight_layout()

# Save the plot as PNG and SVG
plt.savefig("scripts/paper_recreation/viz/sarcasm_bias_datasets_chart.png", dpi=300)
plt.savefig("scripts/paper_recreation/viz/sarcasm_bias_datasets_chart.svg")

# Show the plot
plt.show()
