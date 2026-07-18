import os
import numpy as np
from fem_solver import TetrahedralFEMSolver

def generate_hexahedral_grid(nx, ny, nz, lx=1.0, ly=1.0, lz=1.5):
    """
    Generates a 3D grid of nodes and divides each hexahedral brick into 6 tetrahedrons.
    """
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    z = np.linspace(0, lz, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    elements = []
    # Loop over grid cells and split into tetrahedrons
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # 8 corners of the brick element
                n0 = i * ny * nz + j * nz + k
                n1 = (i + 1) * ny * nz + j * nz + k
                n2 = (i + 1) * (ny + 1) * nz + j * nz + k
                n2 = (i + 1) * ny * nz + (j + 1) * nz + k # correct offset
                
                # Standard indexing for 3D grid corners
                idx = lambda x_i, y_i, z_i: x_i * ny * nz + y_i * nz + z_i
                
                p000 = idx(i, j, k)
                p100 = idx(i+1, j, k)
                p010 = idx(i, j+1, k)
                p110 = idx(i+1, j+1, k)
                p001 = idx(i, j, k+1)
                p101 = idx(i+1, j, k+1)
                p011 = idx(i, j+1, k+1)
                p111 = idx(i+1, j+1, k+1)
                
                # Split brick into 6 tetrahedrons
                elements.append([p000, p100, p110, p111])
                elements.append([p000, p100, p111, p101])
                elements.append([p000, p101, p111, p001])
                elements.append([p000, p110, p010, p111])
                elements.append([p000, p010, p111, p011])
                elements.append([p000, p011, p111, p001])
                
    return nodes, np.array(elements)

def generate_dataset(num_patients=200):
    np.random.seed(42)
    
    # 3D grid: 5x5x6 nodes = 150 nodes, 600 elements
    nx, ny, nz = 5, 5, 6
    nodes, elements = generate_hexahedral_grid(nx, ny, nz)
    
    # Identify fixed nodes (Dirichlet BC: base of the bone, z = 0)
    fixed_nodes = np.where(nodes[:, 2] == 0.0)[0]
    
    # Identify load nodes (Neumann BC: top surface, z = max(z))
    z_max = np.max(nodes[:, 2])
    load_nodes = np.where(nodes[:, 2] == z_max)[0]
    
    output_dir = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage2_Transformer/fem_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating FEM training dataset for {num_patients} patients...")
    
    for p_id in range(num_patients):
        # 1. Randomly sample patient bone quality (Young's Modulus E_bone in 1.5 - 15.0 GPa)
        # D4 ~ 1.5, D3 ~ 5.0, D2 ~ 10.0, D1 ~ 15.0 GPa
        E_bone = np.random.choice([1.5, 5.0, 10.0, 15.0])
        
        # 2. Randomly sample yttrium composition Y in [2.0, 5.0] mol%
        Y = np.random.uniform(2.0, 5.0)
        # Zirconia stiffness decreases modestly with Y stabilizer
        E_zirconia = 210.0 - 8.0 * Y
        
        # Element E-field: lower half (z < 0.75) is bone, upper half is zirconia
        E_field = np.zeros(len(elements))
        for el_idx, elem in enumerate(elements):
            # Compute element centroid z-coordinate
            z_centroid = np.mean(nodes[elem, 2])
            if z_centroid < 0.75:
                E_field[el_idx] = E_bone
            else:
                E_field[el_idx] = E_zirconia
                
        # 3. Randomly sample bite force magnitude and direction
        # Normal bite force 50 - 1500 N
        force_mag = np.random.uniform(100.0, 1200.0)
        # Force direction has a primary downward components with small lateral components
        force_dir = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), -1.0])
        force_dir = force_dir / np.linalg.norm(force_dir)
        
        nodal_forces = np.zeros((len(nodes), 3))
        # Distribute force among top load nodes
        for node in load_nodes:
            nodal_forces[node] = (force_mag / len(load_nodes)) * force_dir
            
        # Run FEM solver
        solver = TetrahedralFEMSolver(nodes, elements, E_field, nu=0.3)
        displacements, von_mises = solver.solve(fixed_nodes, nodal_forces)
        
        # Save training sample
        np.savez_compressed(
            os.path.join(output_dir, f"patient_{p_id}.npz"),
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
        print(f"Generated patient {p_id}/{num_patients}")

if __name__ == "__main__":
    generate_dataset(num_patients=200)
