import pandas as pd
import matplotlib.pyplot as plt

# Load the data
train_stats = pd.read_csv('./outputs/train_stats.txt')
val_stats = pd.read_csv('./outputs/validation_stats.txt')

# Set the figure and axes
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Training vs Validation Metrics', fontsize=22, fontweight='bold')

# Define metrics and pretty colors
metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1_score', 'avg_class_accuracy']
titles = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Average Class Accuracy']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']

# Plotting each metric
for idx, ax in enumerate(axs.flat):
    if idx < len(metrics):
        metric = metrics[idx]
        ax.plot(train_stats['epoch'], train_stats[metric], label='Train', color=colors[idx], linewidth=2.5)
        ax.plot(val_stats['epoch'], val_stats[metric], label='Validation', color=colors[idx], linestyle='--', linewidth=2.5)
        ax.set_title(titles[idx], fontsize=16, fontweight='semibold')
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel(titles[idx], fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', labelsize=12)

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('training_validation_metrics_tile.png', dpi=900)
plt.show()
