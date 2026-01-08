import re
import matplotlib.pyplot as plt

# -------- CONFIG --------
LOG_FILE = "8jan.txt"
OUTPUT_PNG = f"iitd_norm_train_val_accuracy_{LOG_FILE.split('.')[0]}.png"
# ------------------------

epochs = []
train_acc = []
val_acc = []

# Regex patterns
epoch_pat = re.compile(r"=== Epoch:\s*(\d+)\s*===")
train_acc_pat = re.compile(r"Avg Train Acc:\s*([0-9.]+)")
val_acc_pat = re.compile(r"Avg Val Acc:\s*([0-9.]+)")
test_acc_pat = re.compile(r"Test Acc:\s*([0-9.]+)")

test_acc = None

with open(LOG_FILE, "r") as f:
    for line in f:
        line = line.strip()

        epoch_match = epoch_pat.search(line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))
            continue

        train_match = train_acc_pat.search(line)
        if train_match:
            train_acc.append(float(train_match.group(1)))
            continue

        val_match = val_acc_pat.search(line)
        if val_match:
            val_acc.append(float(val_match.group(1)))
            continue

        test_match = test_acc_pat.search(line)
        if test_match:
            test_acc = float(test_match.group(1))

# -------- Sanity Check --------
assert len(epochs) == len(train_acc) == len(val_acc), \
    "Mismatch in parsed epochs and accuracies!"

# -------- Plot --------
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
plt.plot(epochs, val_acc, label="Validation Accuracy", marker='s')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ProtoNet Training on IITD-1 Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ✅ SAVE FIGURE
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")

plt.show()

# -------- Stats --------
print(f"Saved plot to: {OUTPUT_PNG}")
print(f"Total epochs parsed: {len(epochs)}")
print(f"Best Validation Accuracy: {max(val_acc):.4f}")
if test_acc is not None:
    print(f"Test Accuracy: {test_acc:.4f}")
