import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the folder containing your CSVs
history_dir = "C:/Users/Matth/Documents/Leiden University/Project/Histories/Model_1_teq_m_r/"

# Store data for comparison
histories = []
val_losses = []
file_names = []

# Load all history CSVs
for filename in os.listdir(history_dir):
    if filename.endswith(".csv"):
        path = os.path.join(history_dir, filename)
        df = pd.read_csv(path)
        
        if 'val_loss' in df.columns:
            histories.append(df)
            val_losses.append(min(df['val_loss']))
            file_names.append(filename)

# Identify the best model (lowest val_loss)
top_5_indices = sorted(range(len(val_losses)), key=lambda i: val_losses[i])[:5]
top_5_files = [file_names[i] for i in top_5_indices]

# Plotting
plt.figure(figsize=(12, 8))

for i, df in enumerate(histories):
    label = file_names[i].replace(".csv", "")
    import re

    # In your loop where you set `label`:
    match = re.search(r"(model[_\s]?\d+)", label, re.IGNORECASE)
    label = match.group(1).replace("_", " ") if match else label
    is_top_5 = (i in top_5_indices)
    alpha = 1.0 if is_top_5 else 0.3
    lw = 2.5 if is_top_5 else 1.0

    if is_top_5:
        # Plot training loss (with label)
        plt.plot(df['loss'], linestyle='-', alpha=alpha, linewidth=lw,
                 label=f"{label} - train", color=None)

        # Plot validation loss (with label)
        plt.plot(df['val_loss'], linestyle='--', alpha=alpha, linewidth=lw,
                 label=f"{label} - val", color=None)
    else:
        # Plot without label
        plt.plot(df['loss'], linestyle='-', alpha=alpha, linewidth=lw, color='gray')
        plt.plot(df['val_loss'], linestyle='--', alpha=alpha, linewidth=lw, color='lightgray')

plt.title("Model Training and Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(top=8,bottom=1.5)
plt.subplots_adjust(right=0.75)
plt.legend(bbox_to_anchor=(1.25, 1), loc='upper right', borderaxespad=0.)
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(right=0.8)

# Save plot
output_path = os.path.join(history_dir, "all_model_histories.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved combined loss plot with highlight at: {output_path}")
print(f"Best model (lowest val_loss): {best_file}")
