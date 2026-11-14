function plot_results(log_data, corridor, walls, params, plan, mpc, sim_cfg)
% PLOT_RESULTS  Generate all simulation figures.
%
%   plot_results(log_data, corridor, walls, params, plan, mpc, sim_cfg)

    %% Figure 1 — main dashboard
    figure('Position', [30 30 1600 1000], 'Color', 'w');

    % 1. Top-down corridor view
    subplot(2,3,[1,4]);
    draw_corridor(corridor, walls); hold on;
    plot(log_data.state(1,:), log_data.state(2,:), 'b-', 'LineWidth', 2.0);
    plot(log_data.tgt(1,:),   log_data.tgt(2,:),   'r-', 'LineWidth', 1.5);
    plot(log_data.ref(1,:),   log_data.ref(2,:),   'g:', 'LineWidth', 1.0);

    plot(log_data.state(1,1), log_data.state(2,1), 'bo', 'MarkerSize', 12, 'LineWidth', 2);
    plot(log_data.tgt(1,1),   log_data.tgt(2,1),   'rs', 'MarkerSize', 12, 'LineWidth', 2);

    rp_idx = 1 : round(plan.T_replan / sim_cfg.dt) : size(log_data.t, 2);
    plot(log_data.state(1,rp_idx), log_data.state(2,rp_idx), 'b^', ...
         'MarkerSize', 6, 'LineWidth', 1.5);
    plot(log_data.tgt_observed(1,rp_idx), log_data.tgt_observed(2,rp_idx), ...
         'rx', 'MarkerSize', 10, 'LineWidth', 2);

    xlabel('x [m]'); ylabel('y [m]');
    title('T-Corridor: Drone Tracking Target');
    legend('','Drone','Target','Planned ref','Drone start','Target start',...
           'Replan pts','Target obs','Location','best');
    axis equal; grid on;

    % 2. X tracking
    subplot(2,3,2);
    plot(log_data.t, log_data.state(1,:), 'b-', 'LineWidth', 1.2); hold on;
    plot(log_data.t, log_data.tgt(1,:), 'r--');
    plot(log_data.t, log_data.ref(1,:), 'g:');
    ylabel('x [m]'); title('X Position'); grid on;
    legend('Drone','Target','Ref','Location','best');

    % 3. Y tracking
    subplot(2,3,5);
    plot(log_data.t, log_data.state(2,:), 'b-', 'LineWidth', 1.2); hold on;
    plot(log_data.t, log_data.tgt(2,:), 'r--');
    plot(log_data.t, log_data.ref(2,:), 'g:');
    xlabel('Time [s]'); ylabel('y [m]'); title('Y Position'); grid on;

    % 4. Pitch
    subplot(2,3,3);
    plot(log_data.t, rad2deg(log_data.state(3,:)), 'k-', 'LineWidth', 1.2); hold on;
    yline( rad2deg(params.max_tilt), 'r--');
    yline(-rad2deg(params.max_tilt), 'r--');
    ylabel('\theta [deg]'); title('Pitch Angle'); grid on;

    % 5. Motor thrusts
    subplot(2,3,6);
    plot(log_data.t, log_data.u(1,:), '-', 'LineWidth', 1.0); hold on;
    plot(log_data.t, log_data.u(2,:), '-', 'LineWidth', 1.0);
    yline(params.F_max, 'r--'); yline(params.F_min, 'r--');
    xlabel('Time [s]'); ylabel('Thrust [N]'); title('Motor Commands');
    legend('F_1','F_2','Limits','Location','best'); grid on;

    sgtitle('2D Quadcopter — T-Corridor MPC Tracking', ...
            'FontSize', 14, 'FontWeight', 'bold');

    %% Figure 2 — diagnostics
    figure('Position', [100 50 900 400], 'Color', 'w');

    err = vecnorm(log_data.state(1:2,:) - log_data.tgt, 2, 1);

    subplot(1,3,1);
    plot(log_data.t, err, 'b-', 'LineWidth', 1.5);
    xlabel('Time [s]'); ylabel('Error [m]');
    title('Tracking Error'); grid on;

    subplot(1,3,2);
    plot(log_data.t, log_data.wall_dist, 'k-', 'LineWidth', 1.2); hold on;
    yline(params.drone_radius, 'r--', 'Safety margin');
    xlabel('Time [s]'); ylabel('Distance [m]');
    title('Min Distance to Wall'); grid on;

    subplot(1,3,3);
    plot(log_data.solve_time, 'k.', 'MarkerSize', 3);
    xlabel('MPC call #'); ylabel('[ms]');
    title('MPC Solve Time'); grid on;
    yline(mpc.dt_ctrl * 1000, 'r--', 'Budget');

    %% Console summary
    fprintf('\n--- Performance ---\n');
    fprintf('Tracking error — mean: %.2f m, max: %.2f m\n', mean(err), max(err));
    fprintf('Min wall distance:     %.3f m (limit: %.2f m)\n', ...
            min(log_data.wall_dist), params.drone_radius);
    fprintf('Wall violations:       %d samples\n', ...
            sum(log_data.wall_dist < params.drone_radius));
end

%% ========================================================================

function draw_corridor(corridor, walls)
    c = corridor;
    fill([c.vert_x_min c.vert_x_max c.vert_x_max c.vert_x_min], ...
         [c.vert_y_min c.vert_y_min c.horz_y_min c.horz_y_min], ...
         [0.95 0.95 0.95], 'EdgeColor', 'none'); hold on;
    fill([c.vert_x_min c.horz_x_max c.horz_x_max c.vert_x_min], ...
         [c.horz_y_min c.horz_y_min c.horz_y_max c.horz_y_max], ...
         [0.95 0.95 0.95], 'EdgeColor', 'none');
    for i = 1:size(walls, 1)
        plot([walls(i,1) walls(i,3)], [walls(i,2) walls(i,4)], ...
             'k-', 'LineWidth', 2.5);
    end
end
