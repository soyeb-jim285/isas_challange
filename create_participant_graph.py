import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set modern style
plt.style.use('default')
sns.set_palette("Set2")

# Load the metrics data
with open('results/metrics/baseline/metrics.json', 'r') as f:
    baseline_data = json.load(f)

with open('results/metrics/optimized/metrics.json', 'r') as f:
    optimized_data = json.load(f)

# Extract participant data
baseline_participants = baseline_data['per_participant']
optimized_participants = optimized_data['per_participant']

# Create data arrays
participants = [f"Participant {p['participant']}" for p in baseline_participants]
baseline_accuracy = [p['accuracy'] * 100 for p in baseline_participants]  # Convert to percentage
baseline_f1 = [p['f1_weighted'] * 100 for p in baseline_participants]  # Convert to percentage
optimized_accuracy = [p['accuracy'] * 100 for p in optimized_participants]  # Convert to percentage
optimized_f1 = [p['f1_weighted'] * 100 for p in optimized_participants]  # Convert to percentage

# Create figure with better layout
fig, ax = plt.subplots(figsize=(12, 8))

# Set up positions for grouped bars
x = np.arange(len(participants))
width = 0.2

# Define professional colors
colors = {
    'baseline_acc': '#2E86AB',
    'optimized_acc': '#A23B72', 
    'baseline_f1': '#F18F01',
    'optimized_f1': '#C73E1D'
}

# Create bars
bars1 = ax.bar(x - width*1.5, baseline_accuracy, width, label='Baseline Accuracy', 
               color=colors['baseline_acc'], alpha=0.8, edgecolor='white', linewidth=1)
bars2 = ax.bar(x - width/2, optimized_accuracy, width, label='Optimized Accuracy', 
               color=colors['optimized_acc'], alpha=0.8, edgecolor='white', linewidth=1)
bars3 = ax.bar(x + width/2, baseline_f1, width, label='Baseline F1-Score', 
               color=colors['baseline_f1'], alpha=0.8, edgecolor='white', linewidth=1)
bars4 = ax.bar(x + width*1.5, optimized_f1, width, label='Optimized F1-Score', 
               color=colors['optimized_f1'], alpha=0.8, edgecolor='white', linewidth=1)

# Customize the plot
ax.set_xlabel('Participants', fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel('Performance (%)', fontsize=14, fontweight='bold', color='#333333')
ax.set_title('Per-Participant Performance: Accuracy and F1-Score Comparison\nBaseline vs Optimized LSTM Models', 
             fontsize=16, fontweight='bold', color='#333333', pad=20)

ax.set_xticks(x)
ax.set_xticklabels(participants, fontsize=11)
ax.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True, shadow=True)

# Add grid for better readability
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Set y-axis limits
ax.set_ylim(0, 80)

# Add value labels on bars with smaller font
def add_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{value:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

add_labels(bars1, baseline_accuracy)
add_labels(bars2, optimized_accuracy)
add_labels(bars3, baseline_f1)
add_labels(bars4, optimized_f1)

# Customize spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')

# Adjust layout and save
plt.tight_layout()
plt.savefig('results/participant_performance_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

# Create detailed analysis
print("="*80)
print("DETAILED PARTICIPANT PERFORMANCE ANALYSIS")
print("="*80)

print(f"\n{'Participant':<12} {'Baseline':<20} {'Optimized':<20} {'Change':<15}")
print(f"{'':12} {'Acc%':<9} {'F1%':<9} {'Acc%':<9} {'F1%':<9} {'Acc':<7} {'F1':<7}")
print("-"*80)

for i in range(len(participants)):
    acc_change = optimized_accuracy[i] - baseline_accuracy[i]
    f1_change = optimized_f1[i] - baseline_f1[i]
    print(f"{participants[i]:<12} {baseline_accuracy[i]:<9.1f} {baseline_f1[i]:<9.1f} "
          f"{optimized_accuracy[i]:<9.1f} {optimized_f1[i]:<9.1f} "
          f"{acc_change:+7.1f} {f1_change:+7.1f}")

# Calculate statistics
acc_improvements = [optimized_accuracy[i] - baseline_accuracy[i] for i in range(len(participants))]
f1_improvements = [optimized_f1[i] - baseline_f1[i] for i in range(len(participants))]

print(f"\nSUMMARY STATISTICS:")
print(f"Average Accuracy Change: {np.mean(acc_improvements):+.2f}%")
print(f"Average F1-Score Change: {np.mean(f1_improvements):+.2f}%")
print(f"Accuracy Std Dev (Baseline): {np.std(baseline_accuracy):.2f}%")
print(f"Accuracy Std Dev (Optimized): {np.std(optimized_accuracy):.2f}%")
print(f"F1-Score Std Dev (Baseline): {np.std(baseline_f1):.2f}%")
print(f"F1-Score Std Dev (Optimized): {np.std(optimized_f1):.2f}%")

print(f"\nGraph saved as: results/participant_performance_comparison.png")