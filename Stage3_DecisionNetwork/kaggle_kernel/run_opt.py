import os
import numpy as np
import json
from scipy.stats import lognorm, norm

# =====================================================================
# 1. Zirconia Decision Network Definition
# =====================================================================
class ZirconiaDecisionNetwork:
    def __init__(self, w_L=1.0, w_C=0.02, sf=1.2):
        self.w_L = w_L
        self.w_C = w_C
        self.sf = sf
        
        self.Y_vals = np.round(np.arange(2.0, 5.1, 0.1), 2)
        self.F_bins = np.logspace(np.log10(50), np.log10(1500), 11)
        self.F_centers = np.sqrt(self.F_bins[:-1] * self.F_bins[1:])
        self.pH_bins = np.linspace(4.0, 8.0, 9)
        self.pH_centers = 0.5 * (self.pH_bins[:-1] + self.pH_bins[1:])
        
        self.v_bins = np.logspace(-12, -6, 13)
        self.v_centers = np.sqrt(self.v_bins[:-1] * self.v_bins[1:])
        
        self.delta_bins = np.linspace(0, 25, 7)
        self.delta_centers = 0.5 * (self.delta_bins[:-1] + self.delta_bins[1:])

    def fit_patient_telemetry(self, F_profile, pH_profile):
        F_shape, F_loc, F_scale = lognorm.fit(F_profile, floc=0)
        F_cpd = lognorm.cdf(self.F_bins[1:], F_shape, F_loc, F_scale) - lognorm.cdf(self.F_bins[:-1], F_shape, F_loc, F_scale)
        F_cpd = F_cpd / np.sum(F_cpd)
        
        pH_mu, pH_std = norm.fit(pH_profile)
        pH_cpd = norm.cdf(self.pH_bins[1:], pH_mu, pH_std) - norm.cdf(self.pH_bins[:-1], pH_mu, pH_std)
        pH_cpd = pH_cpd / np.sum(pH_cpd)
        
        return F_cpd, pH_cpd

    def _compute_paris_v(self, F, pH, Y):
        A = 1.3e-21 * (1.0 + 0.15 * (Y - 3.0)) * (1.0 + 0.5 * (7.0 - pH))
        n = 22.0 * (1.0 - 0.02 * (Y - 3.0))
        K = 0.5 * F * 1e-6
        K_I0 = 3.5 - 0.5 * (Y - 3.0)
        if K < K_I0:
            return 1e-13
        return A * (K**n)

    def _compute_ltd_drift_prob(self, Y):
        rate = 0.2 + 0.5 * (Y - 2.0)
        mean_drift = rate * 10.0
        drift_std = 2.0
        prob = norm.cdf(self.delta_bins[1:], mean_drift, drift_std) - norm.cdf(self.delta_bins[:-1], mean_drift, drift_std)
        prob = np.clip(prob, 1e-6, None)
        return prob / np.sum(prob)

    def _compute_utility(self, v_val, delta_val, Y):
        K_IC0 = 5.5 - 0.9 * (Y - 3.0)
        K_IC = K_IC0 * (1.0 - 0.012 * delta_val)
        L = K_IC / (v_val * self.sf * 3.1536e7)
        L = np.clip(L, 0.0, 50.0)
        C = 30.0 + 5.0 * Y + 3.0 * (5.0 - Y)**2
        return self.w_L * L - self.w_C * C

    def solve_meu(self, F_cpd, pH_cpd):
        EU = np.zeros(len(self.Y_vals))
        for y_idx, Y in enumerate(self.Y_vals):
            delta_cpd = self._compute_ltd_drift_prob(Y)
            y_utility = 0.0
            for f_idx, F_p in enumerate(F_cpd):
                F = self.F_centers[f_idx]
                for ph_idx, pH_p in enumerate(pH_cpd):
                    pH = self.pH_centers[ph_idx]
                    v_val = self._compute_paris_v(F, pH, Y)
                    for d_idx, d_p in enumerate(delta_cpd):
                        delta = self.delta_centers[d_idx]
                        u_val = self._compute_utility(v_val, delta, Y)
                        y_utility += F_p * pH_p * d_p * u_val
            EU[y_idx] = y_utility
            
        best_idx = np.argmax(EU)
        Y_star = self.Y_vals[best_idx]
        return Y_star, EU[best_idx], EU

# =====================================================================
# 2. Main Optimization Sweep
# =====================================================================
def main():
    np.random.seed(42)
    dn = ZirconiaDecisionNetwork()
    results = []
    
    print("Running decision network optimization for 200 patient archetypes...")
    for idx in range(200):
        # Generate synthetic force magnitude representing patient archetypes
        # normal (700-900 N), bruxer (1200-1500 N), elderly (300-500 N)
        archetype_type = idx % 4
        if archetype_type == 0:
            archetype = "Normal"
            force_mag = np.random.uniform(700.0, 900.0)
            pH_profile = np.random.normal(loc=6.6, scale=0.2, size=1000)
        elif archetype_type == 1:
            archetype = "Bruxer"
            force_mag = np.random.uniform(1200.0, 1500.0)
            pH_profile = np.random.normal(loc=6.3, scale=0.2, size=1000)
        elif archetype_type == 2:
            archetype = "Acidic Diet"
            force_mag = np.random.uniform(700.0, 900.0)
            modes = np.random.choice([6.8, 4.2], p=[0.8, 0.2], size=1000)
            pH_profile = modes + np.random.normal(loc=0, scale=0.15, size=1000)
        else:
            archetype = "Elderly"
            force_mag = np.random.uniform(300.0, 500.0)
            pH_profile = np.random.normal(loc=6.5, scale=0.1, size=1000)
            
        F_profile = np.random.lognormal(mean=np.log(force_mag), sigma=0.15, size=1000)
        
        # Fit and solve
        F_cpd, pH_cpd = dn.fit_patient_telemetry(F_profile, pH_profile)
        Y_star, max_u, EU_curve = dn.solve_meu(F_cpd, pH_cpd)
        
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
        actual_EU_no_ph = EU_curve[np.where(dn.Y_vals == Y_no_ph)[0][0]]
        regret_ph = max_u - actual_EU_no_ph
        
        # Ablation 3: No-LTD (delta = 0 constant)
        dn_no_ltd = ZirconiaDecisionNetwork()
        def mock_ltd(Y):
            p = np.zeros(len(dn_no_ltd.delta_centers))
            p[0] = 1.0
            return p
        dn_no_ltd._compute_ltd_drift_prob = mock_ltd
        Y_no_ltd, _, _ = dn_no_ltd.solve_meu(F_cpd, pH_cpd)
        actual_EU_no_ltd = EU_curve[np.where(dn.Y_vals == Y_no_ltd)[0][0]]
        regret_ltd = max_u - actual_EU_no_ltd
        
        # Ablation 4: Population-average Biting Force (F = 700 N constant)
        pop_F_profile = np.ones(1000) * 700.0
        F_pop_cpd, _ = dn.fit_patient_telemetry(pop_F_profile, pH_profile)
        Y_pop, _, _ = dn.solve_meu(F_pop_cpd, pH_cpd)
        actual_EU_pop = EU_curve[np.where(dn.Y_vals == Y_pop)[0][0]]
        regret_pop = max_u - actual_EU_pop
        
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
    
    with open("optimization_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("\nOptimization Complete! Summary:")
    print(f"  - Avg Regret (Categorical Y): {summary['avg_regret_categorical_Y']:.6f}")
    print(f"  - Avg Regret (No pH): {summary['avg_regret_no_ph']:.6f}")
    print(f"  - Avg Regret (No LTD): {summary['avg_regret_no_ltd']:.6f}")
    print(f"  - Avg Regret (Population Force): {summary['avg_regret_population_force']:.6f}")

if __name__ == "__main__":
    main()
