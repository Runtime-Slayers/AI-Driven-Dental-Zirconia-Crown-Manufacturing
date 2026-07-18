import os
import numpy as np
import json
from decision_network import ZirconiaDecisionNetwork

def run_pipeline_optimization():
    # Directories
    dataset_dir = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage2_Transformer/fem_dataset"
    out_dir = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage3_DecisionNetwork"
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if dataset exists locally (otherwise we'll generate a dummy set for testing)
    patient_files = sorted(glob_glob(os.path.join(dataset_dir, "patient_*.npz")))
    if len(patient_files) == 0:
        print("FEM dataset not found locally. Generating a temporary mock set for verification...")
        from sys import path
        path.append("/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage2_Transformer")
        from generate_fem_dataset import generate_dataset
        generate_dataset(num_patients=200)
        patient_files = sorted(glob_glob(os.path.join(dataset_dir, "patient_*.npz")))
        
    dn = ZirconiaDecisionNetwork()
    
    results = []
    
    print("Running decision network optimization for 5 patients...")
    for idx, f_path in enumerate(patient_files[:5]):
        data = np.load(f_path)
        
        # 1. Load patient specific force mag
        # The forces are time-series force magnitudes in Newtons
        force_mag = data['force_mag']
        # Generate 1000 telemetry points centered around this magnitude (with 15% standard deviation)
        np.random.seed(idx)
        F_profile = np.random.lognormal(mean=np.log(force_mag), sigma=0.15, size=1000)
        
        # 2. Generate patient pH telemetry profile
        # elderly/normal ~ 6.5 mode, acidic young adult ~ 4.5 mode
        E_bone = data['E_bone']
        if E_bone == 1.5:  # low-bite-force archetype (elderly)
            pH_profile = np.random.normal(loc=6.5, scale=0.1, size=1000)
        elif force_mag > 1000.0:  # bruxer
            pH_profile = np.random.normal(loc=6.3, scale=0.2, size=1000)
        else: # Normal or Acidic-diet
            if idx % 3 == 0:  # Acidic archetype
                # Mixture model: 80% baseline at 6.8, 20% meal excursions at 4.2
                modes = np.random.choice([6.8, 4.2], p=[0.8, 0.2], size=1000)
                pH_profile = modes + np.random.normal(loc=0, scale=0.15, size=1000)
            else:
                pH_profile = np.random.normal(loc=6.6, scale=0.2, size=1000)
                
        # Fit distributions and solve MEU
        F_cpd, pH_cpd = dn.fit_patient_telemetry(F_profile, pH_profile)
        Y_star, max_u, EU_curve = dn.solve_meu(F_cpd, pH_cpd)
        
        # Compute Expected Lifetime and Cost at optimal composition
        # Average lifetime is calculated based on the expected value of K_IC / (v * sf)
        
        # --- Run Ablations ---
        # Ablation 1: Categorical Y selection (only 3.0, 4.0, or 5.0)
        cat_grid = [3.0, 4.0, 5.0]
        cat_indices = [np.where(dn.Y_vals == val)[0][0] for val in cat_grid]
        cat_EU = EU_curve[cat_indices]
        Y_cat = cat_grid[np.argmax(cat_EU)]
        regret_cat = max_u - np.max(cat_EU)
        
        # Ablation 2: No-pH (neutral pH = 7.0 constant)
        dn_no_ph = ZirconiaDecisionNetwork()
        neutral_pH_profile = np.ones(1000) * 7.0
        _, pH_neutral_cpd = dn_no_ph.fit_patient_telemetry(F_profile, neutral_pH_profile)
        Y_no_ph, _, _ = dn_no_ph.solve_meu(F_cpd, pH_neutral_cpd)
        # Calculate regret using actual pH CPD
        actual_EU_no_ph = np.interp(Y_no_ph, dn.Y_vals, EU_curve)
        regret_ph = max_u - actual_EU_no_ph
        
        # Ablation 3: No-LTD (delta = 0 constant)
        dn_no_ltd = ZirconiaDecisionNetwork()
        # Override LTD probability distribution to put 100% on delta = 0 bin
        def mock_ltd(Y):
            p = np.zeros(len(dn_no_ltd.delta_centers))
            p[0] = 1.0 # 0% monoclinic
            return p
        dn_no_ltd._compute_ltd_drift_prob = mock_ltd
        Y_no_ltd, _, _ = dn_no_ltd.solve_meu(F_cpd, pH_cpd)
        actual_EU_no_ltd = np.interp(Y_no_ltd, dn.Y_vals, EU_curve)
        regret_ltd = max_u - actual_EU_no_ltd
        
        # Ablation 4: Population-average Biting Force (F = 700 N constant)
        pop_F_profile = np.ones(1000) * 700.0
        F_pop_cpd, _ = dn.fit_patient_telemetry(pop_F_profile, pH_profile)
        Y_pop, _, _ = dn.solve_meu(F_pop_cpd, pH_cpd)
        actual_EU_pop = np.interp(Y_pop, dn.Y_vals, EU_curve)
        regret_pop = max_u - actual_EU_pop
        
        # Archetype name mapping
        archetype = "Normal"
        if E_bone == 1.5:
            archetype = "Elderly"
        elif force_mag > 1000.0:
            archetype = "Bruxer"
        elif idx % 3 == 0:
            archetype = "Acidic Diet"
            
        results.append({
            "patient_id": int(idx),
            "archetype": archetype,
            "force_magnitude_n": float(force_mag),
            "optimal_Y": float(Y_star),
            "max_expected_utility": float(max_u),
            "categorical_Y_choice": float(Y_cat),
            "regret_categorical_Y": float(regret_cat),
            "Y_choice_no_ph": float(Y_no_ph),
            "regret_no_ph": float(regret_ph),
            "Y_choice_no_ltd": float(Y_no_ltd),
            "regret_no_ltd": float(regret_ltd),
            "Y_choice_population_force": float(Y_pop),
            "regret_population_force": float(regret_pop)
        })
        
        if idx % 20 == 0:
            print(f"Optimized patient {idx}/200: Y* = {Y_star} mol%, Archetype = {archetype}")
            
    # Calculate global summary statistics
    summary = {
        "avg_regret_categorical_Y": float(np.mean([r["regret_categorical_Y"] for r in results])),
        "avg_regret_no_ph": float(np.mean([r["regret_no_ph"] for r in results])),
        "avg_regret_no_ltd": float(np.mean([r["regret_no_ltd"] for r in results])),
        "avg_regret_population_force": float(np.mean([r["regret_population_force"] for r in results]))
    }
    
    report = {
        "summary": summary,
        "patients": results
    }
    
    report_path = os.path.join(out_dir, "Extra", "optimization_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nOptimization complete! Summary:")
    print(f"  - Avg Regret (Categorical Y menu): {summary['avg_regret_categorical_Y']:.4f}")
    print(f"  - Avg Regret (Neutral pH assumption): {summary['avg_regret_no_ph']:.4f}")
    print(f"  - Avg Regret (No LTD drift model): {summary['avg_regret_no_ltd']:.4f}")
    print(f"  - Avg Regret (Population force assumption): {summary['avg_regret_population_force']:.4f}")
    print(f"Report saved to {report_path}")

def glob_glob(path):
    import glob
    return glob.glob(path)

if __name__ == "__main__":
    run_pipeline_optimization()
