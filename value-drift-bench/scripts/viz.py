import os
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from inspect_ai.log import list_eval_logs, read_eval_log

def load_model_metadata(yaml_path="models.yaml"):
    try:
        with open(yaml_path, "r") as f:
            models = yaml.safe_load(f)
        # Create a lookup dictionary keyed by the slug
        return {m["slug"]: m for m in models}
    except FileNotFoundError:
        print(f"Warning: {yaml_path} not found. Falling back to raw slugs.")
        return {}

def extract_completed_runs(log_dir="./logs", metadata=None):
    metadata = metadata or {}
    data = []
    
    for log_file in list_eval_logs(log_dir):
        log = read_eval_log(log_file)
        
        # Ensure the run produced samples and was successfully scored
        if log.samples and log.samples[0].score:
            score_dict = log.samples[0].score.value
            
            # This automatically skips API failures or unscored JSON errors
            if isinstance(score_dict, dict):
                slug = log.eval.model.replace("openrouter/", "")
                model_info = metadata.get(slug, {})
                
                row = {
                    "Model": model_info.get("nick", slug),
                    "Group": model_info.get("group", "Unknown"),
                }
                row.update(score_dict)
                data.append(row)
                
    return pd.DataFrame(data)

def visualize_results(df):
    if df.empty:
        print("No completed runs found to visualize.")
        return
        
    print(f"Successfully extracted {len(df)} completed model evaluations.")
    
    # 1. Heatmap: Individual Model Scores
    plt.figure(figsize=(10, 8))
    # Group by model in case any were run multiple times, then average
    heatmap_data = df.drop(columns=["Group"]).groupby("Model").mean()
    
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=1, vmax=10, 
                cbar_kws={'label': 'ValueBench Score'})
    plt.title("Moral Constitutions: ValueBench Scores by Model")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("valuebench_heatmap.png", dpi=300)
    print("Saved heatmap to valuebench_heatmap.png")
    
    # 2. Grouped Bar Chart: Comparing 'frontier_closed' vs 'low_safety' etc.
    df_melted = df.melt(id_vars=["Model", "Group"], 
                        var_name="Dimension", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Dimension", y="Score", hue="Group", 
                errorbar=None, palette="Set2")
    plt.title("Average ValueBench Scores by Model Group")
    plt.ylim(0, 10)
    plt.legend(title="Model Group", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("valuebench_groups.png", dpi=300)
    print("Saved group bar chart to valuebench_groups.png")

if __name__ == "__main__":
    meta = load_model_metadata()
    df_scores = extract_completed_runs(metadata=meta)
    visualize_results(df_scores)