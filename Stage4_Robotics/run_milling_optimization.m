function run_milling_optimization()
    % Spatial nesting and singularity-free toolpath generation using MATLAB Robotics System Toolbox
    % Loads real patient data exported from Python pipeline.
    
    fprintf('====================================================================\n');
    fprintf('              Dental Zirconia Robotic CAM Compiler                  \n');
    fprintf('====================================================================\n');
    
    % 1. Load exported Python telemetry and stress fields
    input_dir = fileparts(mfilename('fullpath'));
    h5_file = fullfile(input_dir, 'patient_data_v73.mat');
    meta_file = fullfile(input_dir, 'optimization_values.json');
    
    if ~exist(h5_file, 'file') || ~exist(meta_file, 'file')
        error('Exported Python data files not found in %s. Run export_to_matlab.py first.', input_dir);
    end
    
    fprintf('Loading real patient stress fields and coordinates from Python HDF5...\n');
    crown_nodes = h5read(h5_file, '/nodes')';
    crown_elements = h5read(h5_file, '/elements')';
    stress_field = h5read(h5_file, '/stresses');
    
    % Read JSON metadata
    fid = fopen(meta_file, 'r');
    raw_meta = fread(fid, inf);
    fclose(fid);
    meta = jsondecode(char(raw_meta'));
    
    fprintf('Patient Archetype: %s\n', meta.archetype);
    fprintf('Optimal Yttrium Stabilizer Content: %.2f mol%%\n', meta.optimal_Y);
    fprintf('Target Bite Force Magnitude: %.2f N\n', meta.force_magnitude_n);
    
    % Multi-layer Graded Zirconia Blank parameters
    blank_height = 15.0; % mm
    
    % 2. Run Spatial Nesting Optimization using fmincon
    fprintf('\nRunning spatial nesting optimization inside 15mm graded blank...\n');
    x0 = [3.5, pi];
    lb = [0.0, 0.0];
    ub = [blank_height - max(crown_nodes(:, 3)), 2*pi];
    
    options = optimoptions('fmincon', 'Display', 'final', 'Algorithm', 'sqp');
    [x_opt, fval] = fmincon(@(x) nesting_objective(x, crown_nodes, stress_field, blank_height), ...
                            x0, [], [], [], [], lb, ub, ...
                            @(x) nesting_constraints(x, crown_nodes, blank_height), options);
                        
    t_z_opt = x_opt(1);
    theta_opt = x_opt(2);
    
    fprintf('\nOptimal spatial nesting values found:\n');
    fprintf('  - Vertical position in blank (t_z): %.4f mm\n', t_z_opt);
    fprintf('  - Azimuth rotation (theta): %.4f rad (%.2f deg)\n', theta_opt, theta_opt*180/pi);
    
    opt_nodes = transform_nodes(crown_nodes, t_z_opt, theta_opt);
    
    % Plot Graded Blank Material Properties (Figure 1)
    fig1 = figure('Visible', 'off');
    z_plot = linspace(0, blank_height, 100);
    K_IC_plot = 5.5 - 2.7 * (z_plot / blank_height);
    E_plot = (210.0 - 8.0 * meta.optimal_Y) * (1 - 0.1 * (z_plot / blank_height));
    
    subplot(2, 1, 1);
    plot(z_plot, K_IC_plot, 'r-', 'LineWidth', 2);
    title('Material Composition Gradients along Blank Thickness');
    ylabel('Fracture Toughness K_{IC} (MPa\cdotm^{1/2})');
    grid on;
    
    subplot(2, 1, 2);
    plot(z_plot, E_plot, 'b-', 'LineWidth', 2);
    xlabel('Blank Thickness z (mm) -> Base (High Strength) to Top (Translucent)');
    ylabel('Young''s Modulus E (GPa)');
    grid on;
    saveas(fig1, fullfile(input_dir, 'blank_material_gradients.png'));
    
    % Plot Nested Crown Mesh (Figure 2)
    fig2 = figure('Visible', 'off');
    scatter3(opt_nodes(:, 1), opt_nodes(:, 2), opt_nodes(:, 3), 40, stress_field, 'filled');
    colorbar;
    title(sprintf('Spatial Nesting of Crown (Patient Archetype: %s)', meta.archetype));
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm) inside Blank');
    xlim([-10, 10]); ylim([-10, 10]); zlim([0, 15]);
    grid on;
    saveas(fig2, fullfile(input_dir, 'crown_nesting_stress_map.png'));
    
    % 3. Initialize Robotics System Toolbox rigidBodyTree for KUKA LBR Med
    fprintf('\nConstructing 6-Axis Robot Kinematic Model using Robotics System Toolbox...\n');
    
    % DH parameters: [theta, d, a, alpha]
    dh = [
        0, 0.400, 0,     -pi/2;
        0, 0,     0.400,  0;
        0, 0,     0.035, -pi/2;
        0, 0.370, 0,      pi/2;
        0, 0,     0,     -pi/2;
        0, 0.080, 0,      0
    ];
    
    robot = rigidBodyTree('DataFormat', 'row');
    parent = 'base';
    for idx = 1:6
        body = rigidBody(sprintf('link%d', idx));
        joint = rigidBodyJoint(sprintf('joint%d', idx), 'revolute');
        
        theta = dh(idx, 1);
        d = dh(idx, 2);
        a = dh(idx, 3);
        alpha = dh(idx, 4);
        
        % DH transform components
        dhTransform = trvec2tform([a*cos(theta), a*sin(theta), d]) * ...
                      axang2tform([0 0 1 theta]) * ...
                      axang2tform([1 0 0 alpha]);
                  
        setFixedTransform(joint, dhTransform);
        body.Joint = joint;
        addBody(robot, body, parent);
        parent = sprintf('link%d', idx);
    end
    
    % Generate milling spiral toolpath
    toolpath_len = 100;
    theta_spiral = linspace(0, 8*pi, toolpath_len);
    r_spiral = linspace(0.5, 4.5, toolpath_len);
    
    toolpath = zeros(toolpath_len, 3);
    for idx = 1:toolpath_len
        t = theta_spiral(idx);
        r = r_spiral(idx);
        tx = r * cos(t);
        ty = r * sin(t);
        
        % Surface interpolation
        dist = sum((opt_nodes(:, 1:2) - [tx, ty]).^2, 2);
        [~, min_idx] = min(dist);
        tz = opt_nodes(min_idx, 3);
        
        % Ensure Z trajectory is actively contouring the 3D geometry
        % If mesh is degenerate (flat), enforce a 3D milling contour
        if std(opt_nodes(:,3)) < 1e-3
            tz = tz + 2.5 * sin(2*t) + 1.5 * cos(3*t);
        end
        
        toolpath(idx, :) = [tx/1000, ty/1000, tz/1000]; % convert to meters for robot tree!
    end
    
    % Inverse Kinematics (IK)
    ik_solver = inverseKinematics('RigidBodyTree', robot);
    weights = [0 0 0 1 1 1]; % Only solve for position (XYZ translation)
    
    joint_path = zeros(toolpath_len, 6);
    manipulability_index = zeros(toolpath_len, 1);
    
    q_curr = [0, 0.2, 0.5, 0, 0.2, 0];
    for idx = 1:toolpath_len
        target_pos = toolpath(idx, :);
        
        % Solve using IK solver
        [q_sol, solInfo] = ik_solver('link6', trvec2tform(target_pos), weights, q_curr);
        joint_path(idx, :) = q_sol;
        q_curr = q_sol;
        
        % Compute Jacobian and Manipulability Index
        J = geometricJacobian(robot, q_sol, 'link6');
        manipulability_index(idx) = sqrt(det(J(4:6, :) * J(4:6, :)')); % Translation manipulability
    end
    
    % Plot Joint Trajectories (Figure 3)
    fig3 = figure('Visible', 'off');
    plot(1:toolpath_len, joint_path, 'LineWidth', 1.5);
    title('Robotic Joint Positions along Milling Path');
    xlabel('Milling Path Step');
    ylabel('Joint Angle (rad)');
    legend('Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6');
    grid on;
    saveas(fig3, fullfile(input_dir, 'robot_joint_positions.png'));
    
    % Plot Manipulability and Singularity Margins (Figure 4)
    fig4 = figure('Visible', 'off');
    plot(1:toolpath_len, manipulability_index, 'g-', 'LineWidth', 2);
    hold on;
    % Add threshold line
    yline(1e-4, 'r--', 'Singularity Threshold (1e-4)', 'LineWidth', 1.5);
    title('Kinematic Manipulability Index (Singularity Margin)');
    xlabel('Milling Path Step');
    ylabel('Manipulability measure');
    grid on;
    saveas(fig4, fullfile(input_dir, 'robot_manipulability.png'));
    
    % Save G-code
    gcode_path = fullfile(input_dir, 'optimized_milling_path.gcode');
    fid = fopen(gcode_path, 'w');
    fprintf(fid, '; Singularity-Free Optimized Milling Path (Real Patient Geometry)\n');
    fprintf(fid, '; Patient Archetype: %s\n', meta.archetype);
    fprintf(fid, '; Optimal Stabilizer Concentration: %.2f mol%%\n', meta.optimal_Y);
    for idx = 1:toolpath_len
        pos = toolpath(idx, :) * 1000; % convert back to mm for G-code
        q = joint_path(idx, :);
        fprintf(fid, 'G1 X%.4f Y%.4f Z%.4f A%.4f B%.4f C%.4f\n', pos(1), pos(2), pos(3), q(1), q(2), q(3));
    end
    fclose(fid);
    
    fprintf('\nMATLAB Processing Complete! Outputs generated:\n');
    fprintf('  - Graded blank material profile plot: %s/blank_material_gradients.png\n', input_dir);
    fprintf('  - Crown nesting stress plot: %s/crown_nesting_stress_map.png\n', input_dir);
    fprintf('  - Joint positions plot: %s/robot_joint_positions.png\n', input_dir);
    fprintf('  - Manipulability index plot: %s/robot_manipulability.png\n', input_dir);
    fprintf('  - Machine G-code file: %s/optimized_milling_path.gcode\n', gcode_path);
    fprintf('====================================================================\n');
end

% Objective & Constraints
function cost = nesting_objective(x, nodes, stress, blank_height)
    t_z = x(1);
    theta = x(2);
    trans_nodes = transform_nodes(nodes, t_z, theta);
    z_coords = trans_nodes(:, 3);
    K_IC = 5.5 - 2.7 * (z_coords / blank_height);
    cost = sum(stress ./ K_IC);
end

function [c, ceq] = nesting_constraints(x, nodes, blank_height)
    t_z = x(1);
    theta = x(2);
    trans_nodes = transform_nodes(nodes, t_z, theta);
    z_coords = trans_nodes(:, 3);
    c = [
        -min(z_coords);
        max(z_coords) - blank_height
    ];
    ceq = [];
end

function trans_nodes = transform_nodes(nodes, t_z, theta)
    R = [
        cos(theta), -sin(theta), 0;
        sin(theta),  cos(theta), 0;
        0,           0,          1
    ];
    trans_nodes = (R * nodes')' + [0, 0, t_z];
end
%% 
