# Redefining variables as the environment was reset
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Updated colors to include a fourth color for the new model
colors = ["#CFE4FF", "#FE7070", "#FFDDA6", "#A0FFA0"]  # Added '#A0FFA0' for GPT-3.5-to-4o-BCT

# Categories (bias types)
categories = ["Suggested Answer", "Distractor Fact", "Positional Bias"]

# Existing data for three models
gpt_4o_mini_data = [31.33, 29.28, 47.67]
gpt_4o_mini_unbiased = [23.19, 21.90, None]  # No unbiased value for Positional Bias
control_values = [29.98, 28.09, 38.83]
bct_values = [23.97, 24.64, 33.00]

# New data for the fourth model
gpt35_to_4o_bct_values = [24.60, 24.37, 43.83]

# Inverting the y-order of the 3 categories
categories_reversed = categories[::-1]

fig, ax = plt.subplots(figsize=(8, 5))  # Increased width to accommodate an extra model
height = 0.15  # Reduced height to fit four bars per category

# Define the number of models
num_models = 4
# Calculate the total shift range based on the number of models
shifts = np.linspace(-height * 1.5, height * 1.5, num_models)

# Plotting horizontal bars for the four models
ax.barh(
    np.arange(len(categories_reversed)) + shifts[0],
    gpt_4o_mini_data[::-1],
    height,
    color=colors[0],
    edgecolor="black",
    label="GPT-4o-mini",
)
ax.barh(
    np.arange(len(categories_reversed)) + shifts[1],
    control_values[::-1],
    height,
    color=colors[1],
    edgecolor="black",
    label="Control",
)
ax.barh(
    np.arange(len(categories_reversed)) + shifts[2],
    bct_values[::-1],
    height,
    color=colors[2],
    edgecolor="black",
    label="BCT",
)
ax.barh(
    np.arange(len(categories_reversed)) + shifts[3],
    gpt35_to_4o_bct_values[::-1],
    height,
    color=colors[3],
    edgecolor="black",
    label="GPT-3.5-to-4o-BCT",
)  # New model

# Adding dashed lines for unbiased averages for GPT-4o-mini, specific to each bias category
for i, unbiased_value in enumerate(gpt_4o_mini_unbiased[::-1]):
    if unbiased_value is not None:
        ax.plot([unbiased_value, unbiased_value], [i - height * 2, i + height * 2], color="black", linestyle="dashed")

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
        Line2D([0], [0], color=colors[3], lw=4, label="GPT-3.5-to-4o-BCT"),  # New legend entry
        Line2D([0], [0], color=colors[2], lw=4, label="BCT"),
        Line2D([0], [0], color=colors[1], lw=4, label="Control"),
        Line2D([0], [0], color=colors[0], lw=4, label="GPT-4o-mini"),
        dashed_line,
    ],
    loc="upper right",
)

# Enhance layout to prevent clipping
plt.tight_layout()

# Save the plot as PNG and SVG
plt.savefig("scripts/paper_recreation/viz/comparison_4o35_bias_chart.png")
plt.savefig("scripts/paper_recreation/viz/comparison_4o35_bias_chart.svg")

# Display the plot
plt.show()
