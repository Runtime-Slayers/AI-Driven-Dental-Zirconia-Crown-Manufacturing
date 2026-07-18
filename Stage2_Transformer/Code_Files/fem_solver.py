import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class TetrahedralFEMSolver:
    def __init__(self, nodes, elements, E_field, nu=0.3):
        """
        nodes: np.array of shape (N, 3) representing 3D node coordinates.
        elements: np.array of shape (M, 4) representing 4-node tetrahedral element connectivity.
        E_field: np.array of shape (M,) representing Young's Modulus for each element (GPa).
        nu: Poisson's ratio (default 0.3).
        """
        self.nodes = nodes.astype(np.float64)
        self.elements = elements.astype(np.int64)
        self.E_field = E_field.astype(np.float64)
        self.nu = nu
        self.num_nodes = len(nodes)
        self.num_elements = len(elements)
        
    def _compute_element_stiffness(self, elem_idx):
        """
        Computes the 12x12 element stiffness matrix for a tetrahedral element.
        """
        elem = self.elements[elem_idx]
        E = self.E_field[elem_idx]
        nu = self.nu
        
        # Node coordinates for the element
        coord = self.nodes[elem] # Shape (4, 3)
        
        # Volume of tetrahedron (Jacobian determinant / 6)
        # Det of [1, x1, y1, z1; 1, x2, y2, z2; 1, x3, y3, z3; 1, x4, y4, z4]
        A = np.ones((4, 4))
        A[:, 1:] = coord
        vol = np.abs(np.linalg.det(A)) / 6.0
        if vol < 1e-12:
            return np.zeros((12, 12)), 0.0
            
        # Shape function derivatives (B matrix)
        # B = [B1, B2, B3, B4] where each Bi is 6x3
        # Inv(A) gives the coefficients of the shape functions
        inv_A = np.linalg.inv(A)
        # Derivatives with respect to x, y, z are in columns 1, 2, 3 of inv_A
        # shape_derivs[i] = [b_i, c_i, d_i]
        shape_derivs = inv_A[1:, :].T # Shape (4, 3)
        
        # Assemble strain-displacement matrix B (6 x 12)
        B = np.zeros((6, 12))
        for i in range(4):
            b, c, d = shape_derivs[i]
            # u_x, u_y, u_z displacements for node i map to columns 3i, 3i+1, 3i+2
            B[0, 3*i]   = b
            B[1, 3*i+1] = c
            B[2, 3*i+2] = d
            B[3, 3*i]   = c
            B[3, 3*i+1] = b
            B[4, 3*i+1] = d
            B[4, 3*i+2] = c
            B[5, 3*i]   = d
            B[5, 3*i+2] = b
            
        # Elasticity matrix D (6 x 6) for linear isotropic material
        factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D = np.zeros((6, 6))
        D[0:3, 0:3] = nu
        np.fill_diagonal(D[0:3, 0:3], 1.0 - nu)
        D[3, 3] = 0.5 - nu
        D[4, 4] = 0.5 - nu
        D[5, 5] = 0.5 - nu
        D = D * factor
        
        # Element stiffness matrix K_e = B^T * D * B * Vol
        K_e = np.dot(B.T, np.dot(D, B)) * vol
        return K_e, vol

    def solve(self, fixed_nodes, nodal_forces):
        """
        fixed_nodes: list/array of node indices where displacement is constrained (u = 0).
        nodal_forces: np.array of shape (N, 3) representing external forces applied to nodes.
        """
        print("Assembling global stiffness matrix...")
        # Sparse matrix coordinate lists
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
            # Map 12x12 K_e to global 3N x 3N positions
            for local_r in range(12):
                node_r = elem[local_r // 3]
                dof_r = 3 * node_r + (local_r % 3)
                for local_c in range(12):
                    node_c = elem[local_c // 3]
                    dof_c = 3 * node_c + (local_c % 3)
                    
                    row_indices.append(dof_r)
                    col_indices.append(dof_c)
                    data_values.append(K_e[local_r, local_c])
                    
        # Create sparse matrix
        K_global = sp.coo_matrix((data_values, (row_indices, col_indices)), shape=(3*self.num_nodes, 3*self.num_nodes)).tocsr()
        
        # Nodal force vector F (shape 3N)
        F_global = nodal_forces.flatten()
        
        # Apply Dirichlet boundary conditions (Penalty method or Row/Col elimination)
        # We will use Row/Col elimination for exact constraints
        free_dofs = np.ones(3 * self.num_nodes, dtype=bool)
        for node in fixed_nodes:
            free_dofs[3*node] = False
            free_dofs[3*node+1] = False
            free_dofs[3*node+2] = False
            
        free_indices = np.where(free_dofs)[0]
        
        # Slice K and F
        K_sub = K_global[free_indices, :][:, free_indices]
        F_sub = F_global[free_indices]
        
        print("Solving linear system...")
        displacements_sub = spla.spsolve(K_sub, F_sub)
        
        # Reconstruct full displacement vector
        displacements = np.zeros(3 * self.num_nodes)
        displacements[free_indices] = displacements_sub
        displacements = displacements.reshape((self.num_nodes, 3))
        
        # Post-process: Compute von Mises stress for each node
        print("Post-processing stress fields...")
        nodal_stresses = np.zeros((self.num_nodes, 6))
        node_counts = np.zeros(self.num_nodes)
        
        for i in range(self.num_elements):
            elem = self.elements[i]
            E = self.E_field[i]
            nu = self.nu
            
            # Element displacements
            u_e = displacements[elem].flatten()
            
            # Recalculate B and D
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
            
            # Strain and Stress
            strain = np.dot(B, u_e)
            stress = np.dot(D, strain)
            
            # Distribute stress to element nodes (averaging)
            for node in elem:
                nodal_stresses[node] += stress
                node_counts[node] += 1
                
        # Average stresses at nodes
        for idx in range(self.num_nodes):
            if node_counts[idx] > 0:
                nodal_stresses[idx] /= node_counts[idx]
                
        # Compute Von Mises stress:
        # vm = sqrt( 0.5 * ((s_xx - s_yy)^2 + (s_yy - s_zz)^2 + (s_zz - s_xx)^2 + 6*(s_xy^2 + s_yz^2 + s_zx^2)) )
        s_xx, s_yy, s_zz = nodal_stresses[:, 0], nodal_stresses[:, 1], nodal_stresses[:, 2]
        s_xy, s_yz, s_zx = nodal_stresses[:, 3], nodal_stresses[:, 4], nodal_stresses[:, 5]
        
        von_mises = np.sqrt(0.5 * (
            (s_xx - s_yy)**2 + (s_yy - s_zz)**2 + (s_zz - s_xx)**2 +
            6 * (s_xy**2 + s_yz**2 + s_zx**2)
        ))
        
        print("FEM solution complete.")
        return displacements, von_mises
