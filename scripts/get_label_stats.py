import pandas as pd

# Load the validated labels
file_path = 'data/labels/action_labels_llm_validated.csv'
print(f"Loading {file_path}...")
df = pd.read_csv(file_path)

# Calculate value counts
counts = df['action'].value_counts()
total = len(df)
percentages = (counts / total) * 100

print("\nLabel Distribution:")
for action, count in counts.items():
    pct = percentages[action]
    print(f"- {action}: {pct:.1f}% ({count:,} windows)")

print(f"\nTotal windows: {total:,}")
