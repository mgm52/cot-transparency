# Redefining variables as the environment was reset
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

colors = ["#CFE4FF", "#FE7070", "#FFDDA6"]
# Categories (bias types)
categories = ["Suggested Answer", "Distractor Fact", "Positional Bias"]

gpt_4o_mini_data = [31.33, 29.28, 47.67]
gpt_4o_mini_unbiased = [23.19, 21.90, None]  # No unbiased value for Positional Bias
control_values = [29.98, 28.09, 38.83]
bct_values = [23.97, 24.64, 33.00]

# Inverting the y-order of the 3 categories
categories_reversed = categories[::-1]

fig, ax = plt.subplots(figsize=(8, 4))
height = 0.2

# Plotting horizontal bars for the three models
ax.barh(
    np.arange(len(categories_reversed)) - height,
    gpt_4o_mini_data[::-1],
    height,
    color=colors[0],
    edgecolor="black",
    label="GPT-4o-mini",
)
ax.barh(
    np.arange(len(categories_reversed)),
    control_values[::-1],
    height,
    color=colors[1],
    edgecolor="black",
    label="Control",
)
ax.barh(
    np.arange(len(categories_reversed)) + height,
    bct_values[::-1],
    height,
    color=colors[2],
    edgecolor="black",
    label="BCT",
)

# Adding dashed lines for unbiased averages for GPT-4o-mini, specific to each bias category
for i, unbiased_value in enumerate(gpt_4o_mini_unbiased[::-1]):
    if unbiased_value is not None:
        ax.plot(
            [unbiased_value, unbiased_value], [i - height - 0.1, i + height + 0.1], color="black", linestyle="dashed"
        )

# Adding dashed line to the legend
dashed_line = Line2D([0], [0], color="black", linestyle="dashed", label="GPT-4o-mini Unbiased")

# Labels and title
ax.set_xlabel("Bias/Inconsistent %")
ax.set_xlim(0, 50)
ax.set_yticks(np.arange(len(categories_reversed)))
ax.set_yticklabels(categories_reversed)

# Add a legend including the dashed line
ax.legend(
    handles=[
        Line2D([0], [0], color=colors[2], lw=4, label="BCT"),
        Line2D([0], [0], color=colors[1], lw=4, label="Control"),
        Line2D([0], [0], color=colors[0], lw=4, label="GPT-4o-mini"),
        dashed_line,
    ]
)

# Save the plot as PNG and SVG
plt.tight_layout()
plt.savefig("scripts/paper_recreation/viz/comparison_bias_chart.png")
plt.savefig("scripts/paper_recreation/viz/comparison_bias_chart.svg")

plt.tight_layout()
plt.show()
