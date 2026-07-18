import numpy as np
from scipy.stats import lognorm, norm

class ZirconiaDecisionNetwork:
    def __init__(self, w_L=1.0, w_C=0.5, sf=1.2):
        self.w_L = w_L
        self.w_C = w_C
        self.sf = sf
        
        # Discretization grids
        self.Y_vals = np.round(np.arange(2.0, 5.1, 0.1), 2)  # 31 grid points
        self.F_bins = np.logspace(np.log10(50), np.log10(1500), 11) # 10 bins
        self.F_centers = np.sqrt(self.F_bins[:-1] * self.F_bins[1:]) # Geometric centers
        self.pH_bins = np.linspace(4.0, 8.0, 9) # 8 bins
        self.pH_centers = 0.5 * (self.pH_bins[:-1] + self.pH_bins[1:])
        
        # v (Crack Growth Velocity) 12 bins log-spaced from 1e-12 to 1e-6
        self.v_bins = np.logspace(-12, -6, 13)
        self.v_centers = np.sqrt(self.v_bins[:-1] * self.v_bins[1:])
        
        # delta (LTD Drift) 6 bins from 0 to 25% monoclinic
        self.delta_bins = np.linspace(0, 25, 7)
        self.delta_centers = 0.5 * (self.delta_bins[:-1] + self.delta_bins[1:])

    def fit_patient_telemetry(self, F_profile, pH_profile):
        """
        Fits patient-specific probability distributions for biting force F and pH.
        """
        # Fit LogNormal to biting force
        F_shape, F_loc, F_scale = lognorm.fit(F_profile, floc=0)
        F_cpd = lognorm.cdf(self.F_bins[1:], F_shape, F_loc, F_scale) - lognorm.cdf(self.F_bins[:-1], F_shape, F_loc, F_scale)
        # Normalize
        F_cpd = F_cpd / np.sum(F_cpd)
        
        # Fit Gaussian to pH
        pH_mu, pH_std = norm.fit(pH_profile)
        pH_cpd = norm.cdf(self.pH_bins[1:], pH_mu, pH_std) - norm.cdf(self.pH_bins[:-1], pH_mu, pH_std)
        # Normalize
        pH_cpd = pH_cpd / np.sum(pH_cpd)
        
        return F_cpd, pH_cpd

    def _compute_paris_v(self, F, pH, Y):
        """
        Computes the Paris law crack growth velocity magnitude (m/s).
        """
        A = 1.3e-21 * (1.0 + 0.15 * (Y - 3.0)) * (1.0 + 0.5 * (7.0 - pH))
        n = 22.0 * (1.0 - 0.02 * (Y - 3.0))
        K = 0.5 * F * 1e-6 # Stress intensity factor (approximate scaling)
        # Threshold K_I0 check: below K_I0 no propagation occurs
        K_I0 = 3.5 - 0.5 * (Y - 3.0)
        if K < K_I0:
            return 1e-13 # Sub-threshold negligible velocity
        return A * (K**n)

    def _compute_ltd_drift_prob(self, Y):
        """
        Computes the CPT vector for LTD drift delta given Y over a 10 year service life.
        """
        # Mean monoclinic fraction drift rate (%/yr)
        rate = 0.2 + 0.5 * (Y - 2.0)
        mean_drift = rate * 10.0 # 10 years drift
        # Assume a Gaussian distribution of realized drift around this mean
        drift_std = 2.0
        prob = norm.cdf(self.delta_bins[1:], mean_drift, drift_std) - norm.cdf(self.delta_bins[:-1], mean_drift, drift_std)
        prob = np.clip(prob, 1e-6, None)
        return prob / np.sum(prob)

    def _compute_utility(self, v_val, delta_val, Y, F_val):
        """
        Computes the utility U = w_L * L - w_C * C, incorporating stochastic K_IC.
        """
        # Fracture toughness K_IC decreasing in Y and delta
        K_IC0 = 5.5 - 0.9 * (Y - 3.0)
        K_IC_mean = K_IC0 * (1.0 - 0.012 * delta_val)
        
        # Add stochasticity in K_IC to spread the lifetime distribution
        np.random.seed(42) # Fixed seed for deterministic utility curves
        K_IC_samples = np.random.normal(K_IC_mean, 0.15 * K_IC_mean, 50)
        
        # Lifetime in years (avoid division by zero)
        L_samples = K_IC_samples / (v_val * self.sf * 3.1536e7) # Convert v from m/s to m/year approx
        L_samples = np.clip(L_samples, 0.0, 25.0) # Reduced cap from 50 to 25 years
        L_expected = np.mean(L_samples)
        
        # Material Cost (USD per blank), quadratic in stabilizer content
        C = 30.0 + 5.0 * Y + 3.0 * (5.0 - Y)**2
        
        # Patient-specific risk premium: Bruxers (high F) penalize lower Y more due to higher stress
        risk_premium = 0.001 * F_val * (5.0 - Y)
        
        return self.w_L * L_expected - self.w_C * C - risk_premium

    def solve_meu(self, F_cpd, pH_cpd):
        """
        Runs exact Variable Elimination to calculate the expected utility for each
        yttrium stabilizer concentration in the 31-point grid, then outputs Y*.
        """
        EU = np.zeros(len(self.Y_vals))
        
        for y_idx, Y in enumerate(self.Y_vals):
            # LTD CPD for this Y (CPT: P(delta | Y))
            delta_cpd = self._compute_ltd_drift_prob(Y)
            
            # Sum Expected Utility over joint distribution P(F) P(pH) P(delta | Y) P(v | F, pH, Y)
            y_utility = 0.0
            for f_idx, F_p in enumerate(F_cpd):
                F = self.F_centers[f_idx]
                for ph_idx, pH_p in enumerate(pH_cpd):
                    pH = self.pH_centers[ph_idx]
                    
                    # Paris-law velocity for this F, pH, Y
                    v_val = self._compute_paris_v(F, pH, Y)
                    
                    # Compute expected utility over delta
                    for d_idx, d_p in enumerate(delta_cpd):
                        delta = self.delta_centers[d_idx]
                        u_val = self._compute_utility(v_val, delta, Y, F)
                        # Joint probability
                        joint_p = F_p * pH_p * d_p
                        y_utility += joint_p * u_val
                        
            EU[y_idx] = y_utility
            
        best_idx = np.argmax(EU)
        Y_star = self.Y_vals[best_idx]
        
        # Continuous refinement around the best grid point
        from scipy.optimize import minimize_scalar
        
        def neg_expected_utility(y_cont):
            y_cont = np.clip(y_cont, 2.0, 5.0)
            d_cpd = self._compute_ltd_drift_prob(y_cont)
            y_ut = 0.0
            for f_i, F_p in enumerate(F_cpd):
                F_v = self.F_centers[f_i]
                for ph_i, pH_p in enumerate(pH_cpd):
                    pH_v = self.pH_centers[ph_i]
                    v_v = self._compute_paris_v(F_v, pH_v, y_cont)
                    for d_i, d_p in enumerate(d_cpd):
                        delta_v = self.delta_centers[d_i]
                        u_v = self._compute_utility(v_v, delta_v, y_cont, F_v)
                        y_ut += F_p * pH_p * d_p * u_v
            return -y_ut
            
        res = minimize_scalar(neg_expected_utility, bounds=(max(2.0, Y_star - 0.2), min(5.0, Y_star + 0.2)), method='bounded')
        Y_star_refined = np.round(res.x, 2)
        max_u_refined = -res.fun
        
        return Y_star_refined, max_u_refined, EU

if __name__ == "__main__":
    # Test decision network run
    dn = ZirconiaDecisionNetwork()
    # Mock patient telemetry
    np.random.seed(42)
    mock_F = np.random.lognormal(mean=np.log(700.0), sigma=0.2, size=1000)
    mock_pH = np.random.normal(loc=6.5, scale=0.3, size=1000)
    
    F_cpd, pH_cpd = dn.fit_patient_telemetry(mock_F, mock_pH)
    Y_star, max_u, EU_curve = dn.solve_meu(F_cpd, pH_cpd)
    print(f"Test Patient Optimal Yttrium Stabilizer: {Y_star} mol%")
    print(f"Max Expected Utility: {max_u:.4f}")
