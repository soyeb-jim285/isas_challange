import json

# Load the metrics data
with open('results/metrics/baseline/metrics.json', 'r') as f:
    baseline_data = json.load(f)

with open('results/metrics/optimized/metrics.json', 'r') as f:
    optimized_data = json.load(f)

# Extract participant data
baseline_participants = baseline_data['per_participant']
optimized_participants = optimized_data['per_participant']

print("=== BASELINE DATA ===")
for p in baseline_participants:
    print(f"Participant {p['participant']}: Accuracy = {p['accuracy']:.4f}, F1 = {p['f1_weighted']:.4f}")

print("\n=== OPTIMIZED DATA ===")
for p in optimized_participants:
    print(f"Participant {p['participant']}: Accuracy = {p['accuracy']:.4f}, F1 = {p['f1_weighted']:.4f}")

print("\n=== TIKZ DATA FORMAT ===")
print("Accuracy data for TikZ:")
print("Baseline: ", end="")
for i, p in enumerate(baseline_participants):
    print(f"({i+1},{p['accuracy']:.4f})", end="")
    if i < len(baseline_participants) - 1:
        print(" ", end="")
print()

print("Optimized: ", end="")
for i, p in enumerate(optimized_participants):
    print(f"({i+1},{p['accuracy']:.4f})", end="")
    if i < len(optimized_participants) - 1:
        print(" ", end="")
print()

print("\nF1-Score data for TikZ:")
print("Baseline: ", end="")
for i, p in enumerate(baseline_participants):
    print(f"({i+1},{p['f1_weighted']:.4f})", end="")
    if i < len(baseline_participants) - 1:
        print(" ", end="")
print()

print("Optimized: ", end="")
for i, p in enumerate(optimized_participants):
    print(f"({i+1},{p['f1_weighted']:.4f})", end="")
    if i < len(optimized_participants) - 1:
        print(" ", end="")
print()