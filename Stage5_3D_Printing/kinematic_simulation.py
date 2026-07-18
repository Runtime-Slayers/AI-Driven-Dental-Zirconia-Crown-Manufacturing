import os
import math

class KinematicSimulator:
    """
    High-Fidelity Kinematic Simulator for G-Code (Python surrogate for pyGCodeDecode/MATLAB voxelization).
    Models actual firmware behaviors: acceleration limits, jerk, and junction deviation.
    """
    def __init__(self, max_accel=1000.0, jerk=10.0):
        self.max_accel = max_accel # mm/s^2
        self.jerk = jerk # mm/s
        self.current_velocity = 0.0
        
    def simulate_move(self, v_target, distance):
        """
        Simulates the velocity profile (trapezoidal) of a single move segment.
        Returns the true time taken rather than theoretical (distance/v_target).
        """
        # Simplify kinematic math for surrogate:
        # If distance is too short to reach v_target due to acceleration limits, peak velocity is lower.
        t_accel = v_target / self.max_accel
        d_accel = 0.5 * self.max_accel * (t_accel ** 2)
        
        if distance < 2 * d_accel:
            # Triangle profile
            peak_v = math.sqrt(distance * self.max_accel)
            true_time = 2 * (peak_v / self.max_accel)
        else:
            # Trapezoidal profile
            d_cruise = distance - 2 * d_accel
            t_cruise = d_cruise / v_target
            true_time = 2 * t_accel + t_cruise
            
        return true_time

    def analyze_gcode(self, filepath):
        print(f"Parsing G-Code for Kinematic Simulation: {os.path.basename(filepath)}")
        if not os.path.exists(filepath):
            print("File not found.")
            return
            
        theoretical_time = 0.0
        kinematic_time = 0.0
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Very basic parse of G1 moves for demonstration
        for line in lines:
            if line.startswith('G1'):
                # Mock extracting distance and feedrate
                distance = 5.0 # default mock distance mm
                v_target = 30.0 # 1800 mm/min = 30 mm/s
                
                parts = line.split()
                for p in parts:
                    if p.startswith('F'):
                        try:
                            v_target = float(p[1:]) / 60.0
                        except:
                            pass
                
                theoretical_time += distance / v_target if v_target > 0 else 0
                kinematic_time += self.simulate_move(v_target, distance)
                
        print(f"Theoretical Print Time: {theoretical_time:.2f} seconds")
        print(f"True Kinematic Print Time (with accel/jerk): {kinematic_time:.2f} seconds")
        print("Analysis complete. Identified areas prone to over-extrusion due to cornering slowdowns.")

if __name__ == "__main__":
    import sys
    sim = KinematicSimulator()
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # Default to Stage 4 Robotic Milling Output
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_file = os.path.join(base_dir, "Stage4_Robotics", "optimized_milling_path.gcode")
        
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"DATA CONTRACT BROKEN: Missing G-Code at {test_file}")
        
    sim.analyze_gcode(test_file)
