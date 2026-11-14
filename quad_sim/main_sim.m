%% main_sim.m — 2D Quadcopter T-Corridor Tracking Simulation
%  ================================================================
%  Modules:
%    setup_params.m      — all tunable parameters
%    build_corridor.m    — T-corridor geometry & wall segments
%    build_target.m      — target trajectory along waypoints
%    plan_trajectory.m   — short-horizon corridor-aware planner
%    solve_mpc.m         — MPC controller (quadprog)
%    drone_dynamics.m    — nonlinear plant model + RK4 integrator
%    plot_results.m      — all plotting
%    utils/              — shared helper functions
%  ================================================================

clear; clc; close all;
addpath('utils');

%% Load parameters
[params, mpc, plan, sim_cfg] = setup_params();

%% Build corridor
[corridor, walls] = build_corridor();

%% Build target trajectory
% Human walking at ~5 km/h
target_speed = 5 / 3.6;  % [m/s]
target_waypoints = [10, 2;      % start in vertical corridor
                    10, 19;     % approach junction
                    10, 20;     % junction center
                    12, 20;     % turn right
                    40, 20];    % continue in horizontal corridor

[tgt_t, tgt_pos] = build_target(target_waypoints, target_speed, sim_cfg.T_final + 10);
target_pos_fn = @(t) interp_target(t, tgt_t, tgt_pos);

%% Initial conditions
state  = [10; 0; 0; 0; 0; 0];   % [x, y, theta, vx, vy, omega]
u_prev = [params.F_hover; params.F_hover];

%% Simulation loop
t = 0;
t_next_replan = 0;
t_next_ctrl   = 0;
planned_traj  = [];
u_applied     = u_prev;
last_known_target = target_pos_fn(0);

% Pre-allocate logs
N_log = ceil(sim_cfg.T_final / sim_cfg.dt) + 1;
log_data.t             = zeros(1, N_log);
log_data.state         = zeros(6, N_log);
log_data.u             = zeros(2, N_log);
log_data.tgt           = zeros(2, N_log);
log_data.ref           = zeros(2, N_log);
log_data.tgt_observed  = zeros(2, N_log);
log_data.solve_time    = [];
log_data.wall_dist     = zeros(1, N_log);
idx = 1;

fprintf('Running T-corridor MPC simulation...\n');
tic;

while t < sim_cfg.T_final

    % ---- Replan every T_replan seconds ----
    if t >= t_next_replan
        last_known_target = target_pos_fn(t);
        dt_obs = 0.01;
        tgt_vel_est = (target_pos_fn(t) - target_pos_fn(max(0, t - dt_obs))) / dt_obs;

        planned_traj = plan_trajectory(t, state, last_known_target, ...
            tgt_vel_est, plan, params, corridor);

        t_next_replan = t + plan.T_replan;
    end

    % ---- MPC at control rate ----
    if t >= t_next_ctrl
        x_ref = build_mpc_reference(t, planned_traj, mpc, params);

        st = tic;
        [u_applied, ~] = solve_mpc(state, u_prev, x_ref, mpc, params, walls);
        log_data.solve_time(end+1) = toc(st) * 1000;

        u_prev  = u_applied;
        t_next_ctrl = t + mpc.dt_ctrl;
    end

    % ---- Log ----
    tgt_now = target_pos_fn(t);
    [ref_pos, ~] = sample_trajectory(t, planned_traj);

    log_data.t(idx)              = t;
    log_data.state(:, idx)       = state;
    log_data.u(:, idx)           = u_applied;
    log_data.tgt(:, idx)         = tgt_now;
    log_data.ref(:, idx)         = ref_pos;
    log_data.tgt_observed(:,idx) = last_known_target;
    log_data.wall_dist(idx)      = min_wall_distance(state(1:2), walls);
    idx = idx + 1;

    % ---- Integrate plant ----
    state = rk4_step(state, u_applied, sim_cfg.dt, params);
    state(6) = clamp(state(6), -params.max_omega, params.max_omega);

    t = t + sim_cfg.dt;
end

elapsed = toc;
idx = idx - 1;
fprintf('Done. Wall time: %.1f s (%.1fx real-time)\n', elapsed, sim_cfg.T_final / elapsed);
fprintf('MPC solve: mean %.1f ms, max %.1f ms\n', ...
    mean(log_data.solve_time), max(log_data.solve_time));

% Trim logs
fnames = fieldnames(log_data);
for i = 1:numel(fnames)
    if size(log_data.(fnames{i}), 2) > idx && ~strcmp(fnames{i}, 'solve_time')
        log_data.(fnames{i}) = log_data.(fnames{i})(:, 1:idx);
    end
end

%% Plot
plot_results(log_data, corridor, walls, params, plan, mpc, sim_cfg);
