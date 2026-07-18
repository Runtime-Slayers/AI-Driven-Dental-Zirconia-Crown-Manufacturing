<p align="center">
  <h1 align="center">🦷 AI-Driven Dental Zirconia Crown Manufacturing</h1>
  <p align="center">
    <strong>End-to-end AI pipeline for personalised dental zirconia crown fabrication</strong><br/>
    <em>From CBCT scan to patient-ready 3D-printed crown — fully automated</em>
  </p>
  <p align="center">
    <a href="#overview">Overview</a> •
    <a href="#workflow">Workflow</a> •
    <a href="#stage-architectures">Stage Architectures</a> •
    <a href="#prerequisites">Prerequisites</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#project-structure">Project Structure</a> •
    <a href="#contributing">Contributing</a> •
    <a href="#license">License</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/MATLAB-R2024b-0076A8?logo=mathworks&logoColor=white" alt="MATLAB" />
    <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" />
    <img src="https://img.shields.io/badge/G--Code-RepRap-orange" alt="G-Code" />
  </p>
</p>

---

## Overview

This repository implements a **five-stage, multi-language AI pipeline** for manufacturing patient-specific dental zirconia crowns. The system ingests raw CBCT (Cone-Beam Computed Tomography) scans and autonomously produces optimised G-code for both CNC robotic milling and 3D printing of functionally graded zirconia implants.

### Key Innovations
- **3D Attention U-Net** with Monte Carlo Dropout for boundary uncertainty quantification
- **Twin-Reservoir Bidirectional Graph Transformer** for real-time FEM stress prediction
- **Bayesian Decision Network** with Variable Elimination for optimal yttria stabiliser selection
- **Singularity-free robotic milling** via MATLAB Robotics System Toolbox inverse kinematics
- **RL-optimised toolpath generation** for thermally aware 3D printing

### Languages & Technologies

| Language | Usage |
|----------|-------|
| **Python** (`.py`) | Deep learning (PyTorch), data preprocessing, decision networks, RL agents, G-code generation |
| **MATLAB** (`.m`) | Robotics kinematics (Robotics System Toolbox), Weibull reliability analysis, spatial nesting optimisation |
| **G-Code** (`.gcode`) | CNC milling paths, 3D printing instructions (RepRap flavour) |
| **JSON** | Configuration, kernel metadata, optimisation results |
| **CSV** | FEM mesh data, patient stress fields, force telemetry |
| **HDF5/MAT** | Cross-platform data exchange between Python ↔ MATLAB |

---

## Workflow

The complete end-to-end pipeline flows through five sequential stages:

```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        A[("CBCT Scan\n(DICOM)")]
        B[("Patient Telemetry\n(Bite Force, pH)")]
    end

    subgraph S1["Stage 1 — Segmentation"]
        direction LR
        C["Preprocessing\n& Normalisation"]
        D["3D Attention U-Net\nwith MC Dropout"]
        E["Uncertainty Map\n& Boundary Extraction"]
        C --> D --> E
    end

    subgraph S2["Stage 2 — Transformer"]
        direction LR
        F["FEM Dataset\nGeneration"]
        G["Twin-Reservoir\nBi-Transformer"]
        H["Von Mises Stress\nPrediction"]
        F --> G --> H
    end

    subgraph S3["Stage 3 — Decision Network"]
        direction LR
        I["Patient-Specific\nDistributions"]
        J["Bayesian Decision\nNetwork (VE)"]
        K["Optimal Yttria\nConcentration Y*"]
        I --> J --> K
    end

    subgraph S4["Stage 4 — Robotics"]
        direction LR
        L["Spatial Nesting\n(fmincon)"]
        M["6-DOF IK Solver\n(KUKA LBR Med)"]
        N["Singularity-Free\nMilling G-Code"]
        L --> M --> N
    end

    subgraph S5["Stage 5 — 3D Printing"]
        direction LR
        O["Kinematic\nSimulation"]
        P["RL Toolpath\nOptimiser (PPO)"]
        Q["Patient-Specific\nG-Code Models"]
        O --> P --> Q
    end

    A --> S1
    B --> S3
    S1 -->|"Segmented Mesh"| S2
    S2 -->|"Stress Fields"| S3
    S3 -->|"Material Params"| S4
    S4 -->|"Milling Path"| S5
    S5 --> R[("🦷 Patient-Ready\nZirconia Crown")]

    style INPUT fill:#1a1a2e,color:#fff,stroke:#e94560
    style S1 fill:#16213e,color:#fff,stroke:#0f3460
    style S2 fill:#1a1a2e,color:#fff,stroke:#e94560
    style S3 fill:#16213e,color:#fff,stroke:#0f3460
    style S4 fill:#1a1a2e,color:#fff,stroke:#e94560
    style S5 fill:#16213e,color:#fff,stroke:#0f3460
    style R fill:#e94560,color:#fff,stroke:#0f3460
```

---

## Stage Architectures

### Stage 1 — 3D Attention U-Net Segmentation

```mermaid
flowchart TB
    subgraph Preprocessing
        P1["DICOM/NIfTI\nIngestion"] --> P2["HU Windowing\n& Normalisation"]
        P2 --> P3["3D Patch\nExtraction"]
    end

    subgraph Encoder["Encoder Path"]
        E1["Conv3D Block\n64 filters"] --> E2["MaxPool3D\n↓2×"]
        E2 --> E3["Conv3D Block\n128 filters"] --> E4["MaxPool3D\n↓2×"]
        E4 --> E5["Conv3D Block\n256 filters"] --> E6["MaxPool3D\n↓2×"]
        E6 --> E7["Conv3D Block\n512 filters\n— Bottleneck —"]
    end

    subgraph Decoder["Decoder Path"]
        D1["ConvTranspose3D\n↑2×"] --> AG1{"Attention\nGate"}
        AG1 --> D2["Conv3D Block\n256 filters"]
        D2 --> D3["ConvTranspose3D\n↑2×"] --> AG2{"Attention\nGate"}
        AG2 --> D4["Conv3D Block\n128 filters"]
        D4 --> D5["ConvTranspose3D\n↑2×"] --> AG3{"Attention\nGate"}
        AG3 --> D6["Conv3D Block\n64 filters"]
    end

    subgraph Output
        O1["1×1×1 Conv3D"] --> O2["Sigmoid\nActivation"]
        O2 --> O3["Segmentation\nMask"]
    end

    subgraph Uncertainty["MC Dropout Inference"]
        MC1["T=50 Forward\nPasses"] --> MC2["Mean\nPrediction"]
        MC1 --> MC3["Variance Map\nEpistemic Uncertainty"]
    end

    subgraph Loss["Focal + Dice Loss"]
        L1["Focal Loss\nalpha=0.8 gamma=2.0"]
        L2["Dice Loss\nsmooth=1e-6"]
    end

    P3 --> E1
    E5 -.->|"Skip Connection"| AG1
    E3 -.->|"Skip Connection"| AG2
    E1 -.->|"Skip Connection"| AG3
    E7 --> D1
    D6 --> O1
    O3 --> MC1

    style Encoder fill:#0d1b2a,color:#fff,stroke:#1b263b
    style Decoder fill:#1b263b,color:#fff,stroke:#415a77
    style Uncertainty fill:#e63946,color:#fff,stroke:#1d3557
    style Output fill:#457b9d,color:#fff,stroke:#1d3557
```

### Stage 2 — Twin-Reservoir Bidirectional Graph Transformer

```mermaid
flowchart TB
    subgraph Input["Dual Input Streams"]
        I1["Node Coordinates\nB N 3"]
        I2["Force Vectors\nB N 3"]
    end

    subgraph TwinRes["Twin Reservoir Encoder"]
        R1["W_geo Fixed Random\ntanh coord × W"]
        R2["W_load Fixed Random\ntanh force × W"]
        R1 --> AE1["Trainable AE\nCompress → D_dim"]
        R2 --> AE2["Trainable AE\nCompress → D_dim"]
        AE1 --> FUSE["Additive\nFusion z_geo + z_load"]
        AE2 --> FUSE
    end

    subgraph GraphDown["GraphSAGE Downsampler"]
        GD1["Linear Projections"]
        GD2["Adaptive Max Pool\n→ 150 Super-Nodes"]
        GD1 --> GD2
    end

    subgraph PosEmb["Fourier Positional Embedding"]
        FP1["Random Fourier\nProjection B∈R 3 D/2"]
        FP2["sin/cos\nConcatenation"]
        FP1 --> FP2
    end

    subgraph Transformer["Bidirectional Transformer"]
        T1["Multi-Head\nSelf-Attention\nh=4"]
        T2["Feed-Forward\nNetwork"]
        T3["Layer Norm\n+ Residual"]
        T1 --> T2 --> T3
        T3 -->|"×3 Layers"| T1
    end

    subgraph Head["Regression Head"]
        H1["Linear D 32\n+ ReLU"]
        H2["Linear 32 1"]
        H3["Interpolate\n↑ N nodes"]
        H1 --> H2 --> H3
    end

    subgraph PhysLoss["Physics-Informed Loss"]
        PL1["MSE Loss"]
        PL2["Equilibrium Constraint\nnabla sigma + b = 0"]
    end

    I1 --> R1
    I2 --> R2
    FUSE --> GraphDown
    I1 --> PosEmb
    GraphDown --> ADD["Add"]
    PosEmb --> ADD
    ADD --> Transformer
    T3 --> Head
    H3 --> PL1
    H3 --> PL2

    style TwinRes fill:#2d6a4f,color:#fff,stroke:#1b4332
    style GraphDown fill:#40916c,color:#fff,stroke:#2d6a4f
    style Transformer fill:#52b788,color:#000,stroke:#2d6a4f
    style PhysLoss fill:#d62828,color:#fff,stroke:#6a040f
```

### Stage 3 — Bayesian Decision Network

```mermaid
flowchart TB
    subgraph PatientData["Patient Telemetry"]
        PD1["Bite Force Profile\n→ LogNormal Fit"]
        PD2["Saliva pH Profile\n→ Gaussian Fit"]
    end

    subgraph BN["Bayesian Network Structure"]
        Y["Decision Node\nY in 2.0 to 5.0 mol pct\n31 grid points"]
        F["Chance Node\nF LogNorm\n10 bins"]
        pH["Chance Node\npH Normal mu sigma\n8 bins"]
        V["Deterministic Node\nv = Paris Law\nA K^n crack velocity"]
        D["Chance Node\ndelta = LTD Drift\nNormal rate 10yr sigma"]
        U["Utility Node\nU = w_L L − w_C C\n− risk_premium"]
    end

    subgraph VE["Variable Elimination"]
        VE1["Sum over\nP F P pH P delta Y"]
        VE2["EU Y =\nSum P_joint U v delta Y F"]
        VE3["Y* = argmax EU Y"]
    end

    subgraph Refine["Continuous Refinement"]
        CR1["scipy.optimize\nminimize_scalar"]
        CR2["Y* refined ±0.2 mol pct"]
    end

    PD1 --> F
    PD2 --> pH
    Y --> V
    F --> V
    pH --> V
    Y --> D
    V --> U
    D --> U
    F --> U
    U --> VE1 --> VE2 --> VE3
    VE3 --> CR1 --> CR2

    style BN fill:#3c096c,color:#fff,stroke:#240046
    style VE fill:#5a189a,color:#fff,stroke:#3c096c
    style Refine fill:#7b2cbf,color:#fff,stroke:#5a189a
```

### Stage 4 — Robotic Milling (MATLAB)

```mermaid
flowchart TB
    subgraph Import["Data Import"]
        IM1["HDF5 Patient Data\nnodes elements stresses"]
        IM2["JSON Metadata\narchetype Y* force"]
    end

    subgraph Nesting["Spatial Nesting Optimisation"]
        N1["Decision Variables\nt_z vertical theta rotation"]
        N2["Objective:\nmin Sum sigma/K_IC z"]
        N3["Constraints:\n0 le z_node le 15mm"]
        N4["fmincon SQP\n→ Optimal Placement"]
        N1 --> N2 --> N4
        N3 --> N4
    end

    subgraph Robot["6-DOF Robot Model"]
        R1["DH Parameters\nKUKA LBR Med"]
        R2["rigidBodyTree\nConstruction"]
        R3["Spiral Toolpath\nGeneration"]
        R1 --> R2 --> R3
    end

    subgraph IK["Inverse Kinematics"]
        IK1["inverseKinematics\nSolver"]
        IK2["Joint Trajectories\nq1 to q6"]
        IK3["Manipulability Index\nsqrt det J JT"]
        IK4["Singularity Check\nmu gt 1e-4"]
        IK1 --> IK2
        IK1 --> IK3 --> IK4
    end

    subgraph Out["Outputs"]
        O1["G-Code\n6-axis positions"]
        O2["Material Gradient\nPlots"]
        O3["Joint Trajectory\nPlots"]
        O4["Manipulability\nPlots"]
    end

    IM1 --> Nesting
    IM2 --> Nesting
    N4 --> Robot
    R3 --> IK
    IK2 --> O1
    IK2 --> O3
    IK3 --> O4
    N4 --> O2

    style Nesting fill:#003049,color:#fff,stroke:#001219
    style Robot fill:#005f73,color:#fff,stroke:#003049
    style IK fill:#0a9396,color:#fff,stroke:#005f73
    style Out fill:#94d2bd,color:#000,stroke:#0a9396
```

### Stage 5 — RL-Optimised 3D Printing

```mermaid
flowchart TB
    subgraph GCode["G-Code Generation"]
        G1["Load Patient Mesh\nHDF5 from Stage 4"]
        G2["Adaptive Layer\nSlicing"]
        G3["RepRap Header\ntemps homing"]
        G1 --> G2 --> G3
    end

    subgraph RL["RL Toolpath Optimiser"]
        RL1["State: Grid Position\n+ Print History"]
        RL2["Action: Move to\nAdjacent Node"]
        RL3["Reward Function L_VF theta:\n+10 extrusion\n-0.5 dist non-extrusion\n-50 thermal violation"]
        RL4["PPO/DQN Agent\nTraining"]
        RL1 --> RL4
        RL2 --> RL4
        RL3 --> RL4
    end

    subgraph KinSim["Kinematic Simulation"]
        K1["Trapezoidal Velocity\nProfile Modelling"]
        K2["Acceleration and\nJerk Limits"]
        K3["True Print Time\nvs Theoretical"]
        K4["Over-extrusion\nDiagnostics"]
        K1 --> K2 --> K3
        K2 --> K4
    end

    subgraph Models["Output G-Code Models"]
        M1["Pre-Surgery\nAnatomical Model"]
        M2["Standard Implant\nBaseline"]
        M3["AI-Optimised\nPatient Implant"]
        M4["Adaptive Slicing\nImplant"]
        M5["Non-Planar Stress-\nAligned SINP"]
        M6["Bioprinted Tissue\nScaffold"]
    end

    G3 --> Models
    RL4 -->|"Optimised Sequence"| Models
    Models --> KinSim

    style RL fill:#6a040f,color:#fff,stroke:#370617
    style KinSim fill:#9d0208,color:#fff,stroke:#6a040f
    style Models fill:#dc2f02,color:#fff,stroke:#9d0208
```

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | >= 3.9 | Core pipeline (Stages 1, 2, 3, 5) |
| **PyTorch** | >= 2.0 | Deep learning models |
| **MATLAB** | >= R2024b | Robotics and reliability analysis (Stage 4) |
| **Git** | >= 2.40 | Version control |

### Python Dependencies

```
torch >= 2.0
numpy >= 1.24
scipy >= 1.10
h5py >= 3.8
matplotlib >= 3.7
```

### MATLAB Toolboxes (Stage 4)

- **Robotics System Toolbox** — `rigidBodyTree`, `inverseKinematics`
- **Optimization Toolbox** — `fmincon` (SQP algorithm)
- **Statistics and Machine Learning Toolbox** — Weibull reliability analysis

### Optional

| Tool | Purpose |
|------|---------|
| **NVIDIA GPU + CUDA** | Accelerated training (Stages 1 and 2) |
| **Kaggle API** | Push kernels for cloud training |
| **h5py** | HDF5 data exchange between Python and MATLAB |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Runtime-Slayers/AI-Driven-Dental-Zirconia-Crown-Manufacturing.git
cd AI-Driven-Dental-Zirconia-Crown-Manufacturing

# 2. Create Python environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install Python dependencies
pip install torch numpy scipy h5py matplotlib

# 4. (Optional) For Kaggle kernel deployment
pip install kaggle
```

---

## Usage

### Run the Full Pipeline

```bash
# Stage 1 — Segmentation
python Stage1_Segmentation/Code_Files/preprocess_data.py
python Stage1_Segmentation/Code_Files/train_segmentation.py

# Stage 2 — Transformer Stress Prediction
python Stage2_Transformer/Code_Files/generate_fem_dataset.py
python Stage2_Transformer/Code_Files/train_transformer.py

# Stage 3 — Decision Network Optimisation
python Stage3_DecisionNetwork/Code_Files/decision_network.py
python Stage3_DecisionNetwork/Code_Files/export_to_matlab.py

# Stage 4 — Robotic Milling (MATLAB)
# Open MATLAB and run:
#   >> run_milling_optimization

# Stage 5 — 3D Printing
python Stage5_3D_Printing/generate_gcode_models.py
python Stage5_3D_Printing/rl_toolpath_generator.py
python Stage5_3D_Printing/kinematic_simulation.py
```

### Kaggle Cloud Training

Each stage includes a `kaggle_kernel/` directory with a self-contained script and `kernel-metadata.json` for one-click deployment:

```bash
cd Stage1_Segmentation/kaggle_kernel
kaggle kernels push
```

---

## Project Structure

```
AI-Driven-Dental-Zirconia-Crown-Manufacturing/
│
├── Stage1_Segmentation/
│   ├── Code_Files/
│   │   ├── model.py                    # 3D Attention U-Net architecture
│   │   ├── train_segmentation.py       # Training loop + MC Dropout inference
│   │   ├── preprocess_data.py          # DICOM → normalised patches
│   │   ├── process_telemetry.py        # Patient bite-force telemetry processor
│   │   └── visualize_uncertainty.py    # Uncertainty map visualisation
│   ├── kaggle_kernel/                  # Kaggle cloud training kernel
│   ├── kaggle_master_kernel/           # Full pipeline kernel
│   └── extract_datasets.py            # Dataset extraction utility
│
├── Stage2_Transformer/
│   ├── Code_Files/
│   │   ├── model.py                    # Twin-Reservoir Bi-Transformer
│   │   ├── train_transformer.py        # Training loop (MSE + physics loss)
│   │   ├── generate_fem_dataset.py     # Synthetic FEM dataset generator
│   │   ├── fem_solver.py              # Simplified FEM solver
│   │   └── train.py                   # Alternative training entry point
│   └── kaggle_kernel/                  # Kaggle deployment kernel
│
├── Stage3_DecisionNetwork/
│   ├── Code_Files/
│   │   ├── decision_network.py         # Bayesian network + VE solver
│   │   ├── run_optimization.py         # Batch optimisation runner
│   │   ├── plot_optimization_results.py # Result visualisation
│   │   └── export_to_matlab.py         # HDF5 + JSON export for Stage 4
│   ├── kaggle_kernel/                  # Kaggle deployment
│   ├── Image_Outputs/                  # Ablation and regret plots
│   └── Outputs/                        # Decision network results (JSON)
│
├── Stage4_Robotics/
│   ├── run_milling_optimization.m      # Main MATLAB entry point
│   ├── analyze_zirconia_reliability.m  # Weibull reliability analysis
│   ├── master_addon_visualizer.m       # Advanced visualisation dashboard
│   ├── novel_addon_visualizations.m    # Research visualisations
│   ├── *.png                           # Output plots
│   └── *.gcode                         # Generated milling G-code
│
├── Stage5_3D_Printing/
│   ├── generate_gcode_models.py        # Patient-specific G-code models
│   ├── rl_toolpath_generator.py        # RL agent (PPO) for toolpath
│   ├── kinematic_simulation.py         # Firmware-aware kinematic sim
│   └── *.gcode                         # Generated G-code files
│
├── Extra/
│   ├── export_models.py                # ONNX model export utility
│   └── export_transformer.py           # Transformer export utility
│
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add your feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Contributions are freely available under the same licence.

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/Runtime-Slayers">Runtime Slayers</a></sub>
</p>
