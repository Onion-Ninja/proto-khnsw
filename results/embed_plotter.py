import matplotlib.pyplot as plt

# Data from Table 5
embed_dim = [16, 32, 64, 128, 256, 512, 1024]

# Accuracy values for each N-way setting
way_10 = [98.95, 99.15, 99.34, 99.41, 99.37, 99.24, 99.26]
way_20 = [99.02, 99.20, 99.12, 99.29, 99.30, 99.18, 99.23]
way_30 = [98.67, 99.13, 99.10, 99.05, 99.22, 99.02, 99.08]

# 1. SET GLOBAL FONT SIZES (Easiest way)
plt.rcParams.update({'font.size': 18}) # Base font size for everything

plt.figure(figsize=(12, 7)) # Increased figure size slightly for larger fonts

# Plotting the three lines
plt.plot(embed_dim, way_10, marker='o', linestyle='-', label='10-way (%)', color='blue')
plt.plot(embed_dim, way_20, marker='s', linestyle='--', label='20-way (%)', color='green')
plt.plot(embed_dim, way_30, marker='^', linestyle='-.', label='30-way (%)', color='red')

# Using a log scale for x-axis as embedding dimensions are powers of 2
plt.xscale('log', base=2)
plt.xticks(embed_dim, labels=embed_dim)

# Adding labels and title
plt.title('Effect of Embedding Dimension on Performance (CASIA-Iris-Thousand, 3-shot)')
plt.xlabel('Embedding Dimension')
plt.ylabel('Accuracy (%)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# # Highlighting the peak for the report
# plt.annotate('Max: 99.41%', xy=(128, 99.41), xytext=(150, 99.45),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.tight_layout()
# Save the plot as a high-resolution PNG
plt.savefig('embedding_dimension_performance.png', dpi=300, bbox_inches='tight')