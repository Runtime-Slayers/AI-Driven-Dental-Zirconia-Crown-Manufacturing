import os
import glob
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. 3D Tetrahedral FEM Solver
# =====================================================================
class TetrahedralFEMSolver:
    def __init__(self, nodes, elements, E_field, nu=0.3):
        self.nodes = nodes.astype(np.float64)
        self.elements = elements.astype(np.int64)
        self.E_field = E_field.astype(np.float64)
        self.nu = nu
        self.num_nodes = len(nodes)
        self.num_elements = len(elements)
        
    def _compute_element_stiffness(self, elem_idx):
        elem = self.elements[elem_idx]
        E = self.E_field[elem_idx]
        nu = self.nu
        
        coord = self.nodes[elem]
        
        A = np.ones((4, 4))
        A[:, 1:] = coord
        vol = np.abs(np.linalg.det(A)) / 6.0
        if vol < 1e-12:
            return np.zeros((12, 12)), 0.0
            
        inv_A = np.linalg.inv(A)
        shape_derivs = inv_A[1:, :].T
        
        B = np.zeros((6, 12))
        for i in range(4):
            b, c, d = shape_derivs[i]
            B[0, 3*i]   = b
            B[1, 3*i+1] = c
            B[2, 3*i+2] = d
            B[3, 3*i]   = c
            B[3, 3*i+1] = b
            B[4, 3*i+1] = d
            B[4, 3*i+2] = c
            B[5, 3*i]   = d
            B[5, 3*i+2] = b
            
        factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D = np.zeros((6, 6))
        D[0:3, 0:3] = nu
        np.fill_diagonal(D[0:3, 0:3], 1.0 - nu)
        D[3, 3] = 0.5 - nu
        D[4, 4] = 0.5 - nu
        D[5, 5] = 0.5 - nu
        D = D * factor
        
        K_e = np.dot(B.T, np.dot(D, B)) * vol
        return K_e, vol

    def solve(self, fixed_nodes, nodal_forces):
        row_indices = []
        col_indices = []
        data_values = []
        
        volumes = []
        for i in range(self.num_elements):
            K_e, vol = self._compute_element_stiffness(i)
            volumes.append(vol)
            if vol < 1e-12:
                continue
                
            elem = self.elements[i]
            for local_r in range(12):
                node_r = elem[local_r // 3]
                dof_r = 3 * node_r + (local_r % 3)
                for local_c in range(12):
                    node_c = elem[local_c // 3]
                    dof_c = 3 * node_c + (local_c % 3)
                    
                    row_indices.append(dof_r)
                    col_indices.append(dof_c)
                    data_values.append(K_e[local_r, local_c])
                    
        K_global = sp.coo_matrix((data_values, (row_indices, col_indices)), shape=(3*self.num_nodes, 3*self.num_nodes)).tocsr()
        F_global = nodal_forces.flatten()
        
        free_dofs = np.ones(3 * self.num_nodes, dtype=bool)
        for node in fixed_nodes:
            free_dofs[3*node] = False
            free_dofs[3*node+1] = False
            free_dofs[3*node+2] = False
            
        free_indices = np.where(free_dofs)[0]
        K_sub = K_global[free_indices, :][:, free_indices]
        F_sub = F_global[free_indices]
        
        displacements_sub = spla.spsolve(K_sub, F_sub)
        
        displacements = np.zeros(3 * self.num_nodes)
        displacements[free_indices] = displacements_sub
        displacements = displacements.reshape((self.num_nodes, 3))
        
        nodal_stresses = np.zeros((self.num_nodes, 6))
        node_counts = np.zeros(self.num_nodes)
        
        for i in range(self.num_elements):
            elem = self.elements[i]
            E = self.E_field[i]
            nu = self.nu
            
            u_e = displacements[elem].flatten()
            
            A = np.ones((4, 4))
            A[:, 1:] = self.nodes[elem]
            inv_A = np.linalg.inv(A)
            shape_derivs = inv_A[1:, :].T
            
            B = np.zeros((6, 12))
            for k in range(4):
                b, c, d = shape_derivs[k]
                B[0, 3*k]   = b
                B[1, 3*k+1] = c
                B[2, 3*k+2] = d
                B[3, 3*k]   = c
                B[3, 3*k+1] = b
                B[4, 3*k+1] = d
                B[4, 3*k+2] = c
                B[5, 3*k]   = d
                B[5, 3*k+2] = b
                
            factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
            D = np.zeros((6, 6))
            D[0:3, 0:3] = nu
            np.fill_diagonal(D[0:3, 0:3], 1.0 - nu)
            D[3, 3] = 0.5 - nu
            D[4, 4] = 0.5 - nu
            D[5, 5] = 0.5 - nu
            D = D * factor
            
            strain = np.dot(B, u_e)
            stress = np.dot(D, strain)
            
            for node in elem:
                nodal_stresses[node] += stress
                node_counts[node] += 1
                
        for idx in range(self.num_nodes):
            if node_counts[idx] > 0:
                nodal_stresses[idx] /= node_counts[idx]
                
        s_xx, s_yy, s_zz = nodal_stresses[:, 0], nodal_stresses[:, 1], nodal_stresses[:, 2]
        s_xy, s_yz, s_zx = nodal_stresses[:, 3], nodal_stresses[:, 4], nodal_stresses[:, 5]
        
        von_mises = np.sqrt(0.5 * (
            (s_xx - s_yy)**2 + (s_yy - s_zz)**2 + (s_zz - s_xx)**2 +
            6 * (s_xy**2 + s_yz**2 + s_zx**2)
        ))
        
        return displacements, von_mises

# =====================================================================
# 2. Data Generation
# =====================================================================
def generate_hexahedral_grid(nx, ny, nz, lx=1.0, ly=1.0, lz=1.5):
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    z = np.linspace(0, lz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    elements = []
    idx = lambda x_i, y_i, z_i: x_i * ny * nz + y_i * nz + z_i
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                p000 = idx(i, j, k)
                p100 = idx(i+1, j, k)
                p010 = idx(i, j+1, k)
                p110 = idx(i+1, j+1, k)
                p001 = idx(i, j, k+1)
                p101 = idx(i+1, j, k+1)
                p011 = idx(i, j+1, k+1)
                p111 = idx(i+1, j+1, k+1)
                
                elements.append([p000, p100, p110, p111])
                elements.append([p000, p100, p111, p101])
                elements.append([p000, p101, p111, p001])
                elements.append([p000, p110, p010, p111])
                elements.append([p000, p010, p111, p011])
                elements.append([p000, p011, p111, p001])
    return nodes, np.array(elements)

def generate_fem_dataset(dataset_dir, num_patients=200):
    os.makedirs(dataset_dir, exist_ok=True)
    nx, ny, nz = 5, 5, 6
    nodes, elements = generate_hexahedral_grid(nx, ny, nz)
    fixed_nodes = np.where(nodes[:, 2] == 0.0)[0]
    z_max = np.max(nodes[:, 2])
    load_nodes = np.where(nodes[:, 2] == z_max)[0]
    
    print("Generating synthetic patient FEM data on Kaggle...")
    for p_id in range(num_patients):
        E_bone = np.random.choice([1.5, 5.0, 10.0, 15.0])
        Y = np.random.uniform(2.0, 5.0)
        E_zirconia = 210.0 - 8.0 * Y
        
        E_field = np.zeros(len(elements))
        for el_idx, elem in enumerate(elements):
            z_centroid = np.mean(nodes[elem, 2])
            if z_centroid < 0.75:
                E_field[el_idx] = E_bone
            else:
                E_field[el_idx] = E_zirconia
                
        force_mag = np.random.uniform(100.0, 1200.0)
        force_dir = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), -1.0])
        force_dir = force_dir / np.linalg.norm(force_dir)
        
        nodal_forces = np.zeros((len(nodes), 3))
        for node in load_nodes:
            nodal_forces[node] = (force_mag / len(load_nodes)) * force_dir
            
        solver = TetrahedralFEMSolver(nodes, elements, E_field, nu=0.3)
        displacements, von_mises = solver.solve(fixed_nodes, nodal_forces)
        
        np.savez_compressed(
            os.path.join(dataset_dir, f"patient_{p_id}.npz"),
            nodes=nodes,
            elements=elements,
            E_field=E_field,
            forces=nodal_forces,
            displacements=displacements,
            von_mises=von_mises,
            Y=Y,
            E_bone=E_bone,
            force_mag=force_mag
        )

# =====================================================================
# 3. Model Definition
# =====================================================================
class TwinReservoirEncoder(nn.Module):
    def __init__(self, R_dim=256, D_dim=64):
        super(TwinReservoirEncoder, self).__init__()
        self.R_dim = R_dim
        self.D_dim = D_dim
        self.register_buffer("W_geo", torch.randn(3, R_dim))
        self.register_buffer("W_load", torch.randn(3, R_dim))
        
        self.compress_geo = nn.Sequential(
            nn.Linear(R_dim, D_dim),
            nn.ReLU(),
            nn.Linear(D_dim, D_dim)
        )
        self.compress_load = nn.Sequential(
            nn.Linear(R_dim, D_dim),
            nn.ReLU(),
            nn.Linear(D_dim, D_dim)
        )
        
    def forward(self, nodes_coord, forces):
        h_geo = torch.tanh(torch.matmul(nodes_coord, self.W_geo))
        h_load = torch.tanh(torch.matmul(forces, self.W_load))
        z_geo = self.compress_geo(h_geo)
        z_load = self.compress_load(h_load)
        return z_geo + z_load

class BidirectionalGraphTransformer(nn.Module):
    def __init__(self, D_dim=64, num_heads=4, num_layers=3):
        super(BidirectionalGraphTransformer, self).__init__()
        self.reservoir_encoder = TwinReservoirEncoder(R_dim=256, D_dim=D_dim)
        self.pos_embedding = nn.Linear(3, D_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_dim,
            nhead=num_heads,
            dim_feedforward=2*D_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.stress_head = nn.Sequential(
            nn.Linear(D_dim, D_dim // 2),
            nn.ReLU(),
            nn.Linear(D_dim // 2, 1)
        )
        
    def forward(self, nodes_coord, forces):
        z_combined = self.reservoir_encoder(nodes_coord, forces)
        pos = self.pos_embedding(nodes_coord)
        x = z_combined + pos
        features = self.transformer_encoder(x)
        stress_pred = self.stress_head(features)
        return stress_pred.squeeze(-1)

# Custom Dataset
class BiomechanicalFEMDataset(Dataset):
    def __init__(self, dataset_dir):
        self.files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        nodes = data['nodes'].astype(np.float32)
        forces = data['forces'].astype(np.float32)
        von_mises = data['von_mises'].astype(np.float32)
        return torch.from_numpy(nodes), torch.from_numpy(forces), torch.from_numpy(von_mises)

# =====================================================================
# 4. Training Loop
# =====================================================================
def main():
    dataset_dir = "fem_dataset"
    generate_fem_dataset(dataset_dir, num_patients=200)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = BiomechanicalFEMDataset(dataset_dir)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = BidirectionalGraphTransformer(D_dim=64, num_heads=4, num_layers=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    epochs = 30
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for nodes, forces, targets in train_loader:
            nodes, forces, targets = nodes.to(device), forces.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(nodes, forces)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * nodes.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for nodes, forces, targets in val_loader:
                nodes, forces, targets = nodes.to(device), forces.to(device), targets.to(device)
                outputs = model(nodes, forces)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * nodes.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "transformer_stress.pth")
            print("Saved best transformer weights.")

if __name__ == "__main__":
    main()
