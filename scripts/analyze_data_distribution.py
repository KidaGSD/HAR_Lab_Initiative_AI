import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze():
    # Load Data
    df_scenarios = pd.read_csv("data/labels/scenario_labels.csv")
    df_actions = pd.read_csv("data/labels/master_annotations.csv")
    
    print("=== 1. Scenario Distribution by Split ===")
    # Crosstab of Scenario vs Split
    split_dist = pd.crosstab(df_scenarios['scenario'], df_scenarios['split'])
    print(split_dist)
    
    print("\n=== 2. Low-Level Label Coverage ===")
    # Check which scenarios have LL labels
    ll_counts = df_actions['scenario'].value_counts()
    hl_counts = df_scenarios['scenario'].value_counts()
    
    coverage = pd.DataFrame({'Total_Videos': hl_counts, 'Videos_with_LL': ll_counts}).fillna(0)
    coverage['Coverage_Ratio'] = coverage['Videos_with_LL'] / coverage['Total_Videos']
    print(coverage.sort_values('Coverage_Ratio', ascending=False))
    
    print("\n=== 3. Action Class Balance ===")
    print(df_actions['action'].value_counts(normalize=True))

if __name__ == "__main__":
    analyze()
