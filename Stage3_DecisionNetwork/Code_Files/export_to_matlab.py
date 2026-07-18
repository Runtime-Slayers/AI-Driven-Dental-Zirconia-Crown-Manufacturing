import os
import json
import numpy as np
import h5py

def export_data():
    report_path = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage3_DecisionNetwork/Extra/optimization_report.json"
    patient_dir = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage2_Transformer/fem_dataset"
    out_dir = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage4_Robotics"
    
    if not os.path.exists(report_path):
        print(f"Error: Report not found at {report_path}")
        return
        
    # Load report
    with open(report_path, "r") as f:
        report = json.load(f)
        
    # Get Patient 0 data
    patient_0_rec = report["patients"][0]
    opt_Y = patient_0_rec["optimal_Y"]
    
    # Load patient 0 geometry
    p0_file = os.path.join(patient_dir, "patient_0.npz")
    if not os.path.exists(p0_file):
        print(f"Error: Patient 0 geometry not found at {p0_file}")
        return
        
    data = np.load(p0_file)
    nodes = data["nodes"] * 10.0 # scale to mm
    elements = data["elements"] + 1 # Convert to 1-based MATLAB index
    stresses = data["von_mises"]
    
    # Save HDF5 / v7.3 MAT file
    h5_path = os.path.join(out_dir, "patient_data_v73.mat")
    with h5py.File(h5_path, 'w') as f:
        # chunking prevents memory overflow during MATLAB import
        f.create_dataset('nodes', data=nodes, dtype=np.double, chunks=True, compression="gzip")
        f.create_dataset('elements', data=elements, dtype=np.int32, chunks=True, compression="gzip")
        f.create_dataset('stresses', data=stresses, dtype=np.double, chunks=True, compression="gzip")
    
    # Save JSON config
    meta = {
        "optimal_Y": float(opt_Y),
        "archetype": patient_0_rec["archetype"],
        "force_magnitude_n": float(patient_0_rec["force_magnitude_n"])
    }
    with open(os.path.join(out_dir, "optimization_values.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    print(f"Successfully exported lossless HDF5 (MATLAB v7.3) data to {h5_path}.")

if __name__ == "__main__":
    export_data()
