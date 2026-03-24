import matplotlib.pyplot as plt

# Shared N-way values
ways = [5, 10, 15, 20, 30, 50, 100, 150, 200]

# Helper function to convert 0-1 range to 0-100%
def to_perc(data): 
    return [x * 100 if x <= 1.0 else x for x in data]

# ---------------------------------------------------------
# FIGURE 1: CASIA to IITD
# ---------------------------------------------------------
val_c_to_i = [0.9960, 0.9954, 0.9930, 0.9920, 0.9938, 0.9911, 0.9890, 0.9886, 0.9874]
test_c_to_i = [0.9994, 0.9996, 0.9997, 0.9995, 0.9991, 0.9990, 0.9993, 0.9991, 0.9991]

plt.figure(figsize=(8, 5))
plt.plot(ways, to_perc(val_c_to_i), marker='o', linestyle='--', color='#2E7D32', label='Validation Accuracy')
plt.plot(ways, to_perc(test_c_to_i), marker='s', linestyle='-', color='#1B5E20', label='Test Accuracy', linewidth=2)

plt.title('Scalability Analysis (Train: CASIA-1000, Test: IITD)', fontsize=12, fontweight='bold')
plt.xlabel('Number of Classes (N-way)')
plt.ylabel('Accuracy (%)')
plt.ylim(97.5, 100.2)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('c_to_i.png', dpi=300)
plt.show()

# ---------------------------------------------------------
# FIGURE 2: IITD to CASIA
# ---------------------------------------------------------
val_i_to_c = [100, 100, 100, 100, 100, 100, 100, 100, 100]
test_i_to_c = [0.9840, 0.9757, 0.9417, 0.9700, 0.9647, 0.9541, 0.9483, 0.9491, 0.9518]

plt.figure(figsize=(8, 5))
plt.plot(ways, to_perc(val_i_to_c), marker='o', linestyle='--', color='#1565C0', label='Validation Accuracy')
plt.plot(ways, to_perc(test_i_to_c), marker='s', linestyle='-', color='#0D47A1', label='Test Accuracy', linewidth=2)

plt.title('Scalability Analysis (Train: IITD, Test: CASIA-1000)', fontsize=12, fontweight='bold')
plt.xlabel('Number of Classes (N-way)')
plt.ylabel('Accuracy (%)')
plt.ylim(92.0, 100.2)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('i_to_c.png', dpi=300)
plt.show()