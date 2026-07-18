import os
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_optimization_plots():
    report_path = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage3_DecisionNetwork/optimization_report.json"
    if not os.path.exists(report_path):
        print(f"Report not found at {report_path}")
        return
        
    with open(report_path, "r") as f:
        report = json.load(f)
        
    patients = report["patients"]
    summary = report["summary"]
    
    # 1. Distribution of Optimal Y
    Y_opt = [p["optimal_Y"] for p in patients]
    archetypes = [p["archetype"] for p in patients]
    
    unique_archs = sorted(list(set(archetypes)))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for idx, arch in enumerate(unique_archs):
        arch_Y = [Y_opt[i] for i in range(len(Y_opt)) if archetypes[i] == arch]
        plt.hist(arch_Y, bins=np.arange(2.0, 5.2, 0.1), alpha=0.6, label=arch, color=colors[idx], edgecolor='black')
        
    plt.title("Distribution of Optimal Yttrium mol% Stabilizer")
    plt.xlabel("Yttrium Stabilizer Content (mol%)")
    plt.ylabel("Number of Patients")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Regret Comparison of Ablations
    ablations = ["Categorical Menu", "Neutral pH", "No LTD Drift", "Population Force"]
    regret_vals = [
        summary["avg_regret_categorical_Y"],
        summary["avg_regret_no_ph"],
        summary["avg_regret_no_ltd"],
        summary["avg_regret_population_force"]
    ]
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(ablations, regret_vals, color='#4A90E2', edgecolor='black', width=0.6)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
        
    plt.title("Expected Regret Comparison across Model Ablations")
    plt.ylabel("Average Utility Loss (Regret)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage3_DecisionNetwork/ablation_regret_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Ablation regret plot successfully saved to {plot_path}")

if __name__ == "__main__":
    generate_optimization_plots()
