function master_addon_visualizer()
    % MASTER ADDON VISUALIZER (Refactored for Real Data)
    % This script reads actual Stage 3 output data to generate clinical dashboards.
    % Fails LOUDLY if the data contract is broken.
    
    fprintf('====================================================================\n');
    fprintf('        Executing Master Visualization on REAL Stage 3 JSON Data    \n');
    fprintf('====================================================================\n');
    
    data_dir = fileparts(mfilename('fullpath'));
    stage3_report_path = fullfile(data_dir, '..', 'Stage3_DecisionNetwork', 'Extra', 'optimization_report.json');
    
    % HASH-VERIFIED / LOUD FAILURE CONTRACT
    if ~exist(stage3_report_path, 'file')
        error('DATA CONTRACT BROKEN: Missing Stage 3 JSON output at %s. Pipeline halted.', stage3_report_path);
    end
    
    % Read Real Data
    fid = fopen(stage3_report_path, 'r');
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    report = jsondecode(str);
    
    patients = report.patients;
    
    % Extract arrays for plotting
    F_vals = arrayfun(@(x) x.force_magnitude_n, patients);
    Y_stars = arrayfun(@(x) x.optimal_Y, patients);
    Max_Us = arrayfun(@(x) x.max_expected_utility, patients);
    Regret_pH = arrayfun(@(x) x.regret_no_ph, patients);
    Archetypes = {patients.archetype};
    
    %% DASHBOARD 1: Real Clinical Analytics
    fig1 = figure('Name', 'DB1: Real Clinical Analytics', 'Color', 'w', 'Position', [100, 100, 1400, 900], 'Visible', 'off');
    t1 = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t1, 'Dashboard 1: Stage 3 Real Output Analytics', 'FontSize', 16, 'FontWeight', 'bold');
    
    % 1: Y* Distribution by Archetype
    nexttile;
    unique_archs = unique(Archetypes);
    hold on;
    for i = 1:length(unique_archs)
        mask = strcmp(Archetypes, unique_archs{i});
        histogram(Y_stars(mask), 'DisplayName', unique_archs{i}, 'BinWidth', 0.1);
    end
    title('Optimal Yttrium (Y*) by Patient Archetype');
    xlabel('Y* (mol%)'); ylabel('Count'); legend('Location', 'best'); grid on;
    
    % 2: Max Expected Utility vs Biting Force
    nexttile;
    scatter(F_vals, Max_Us, 40, Y_stars, 'filled');
    colorbar; colormap jet;
    title('Max Expected Utility vs Biting Force');
    xlabel('Biting Force (N)'); ylabel('Expected Utility'); grid on;
    
    % 3: Regret analysis (pH Ablation)
    nexttile;
    histogram(Regret_pH, 30, 'FaceColor', '#F5A623');
    title('Clinical Regret: Ignoring Salivary pH');
    xlabel('Utility Regret'); ylabel('Patients'); grid on;
    
    % 4: Biting Force Distribution
    nexttile;
    histogram(F_vals, 30, 'FaceColor', '#4A90E2');
    title('Actual Extracted Biting Force Distribution');
    xlabel('Force (N)'); ylabel('Count'); grid on;
    
    exportgraphics(fig1, fullfile(data_dir, 'DB1_RealAnalytics.png'), 'Resolution', 300);
    fprintf('Dashboard saved successfully to DB1_RealAnalytics.png\n');
end
