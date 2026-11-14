function [params, mpc, plan, sim_cfg] = setup_params()
% SETUP_PARAMS  All tunable parameters for the 2D quadcopter simulation.
%
%   [params, mpc, plan, sim_cfg] = setup_params()
%
%   params   — drone physical parameters & constraints
%   mpc      — MPC controller settings & weights
%   plan     — short-horizon planner settings
%   sim_cfg  — simulation timestep & duration
%
%   Hardware reference:
%     Motors:  iFlight XING2 1404 4600KV (unibell)
%     Battery: 4S Samsung 50S 21700 (16.8V max, 14.4V nom, 35A cont.)
%     Props:   4×3  (assumed — max thrust ~574g/motor at 4S)
%     Frame:   30 cm × 40 cm motor-to-motor
%     AUW:     686 g

    %% Drone physical parameters
    params.m       = 0.686;     % all-up weight [kg]
    params.g       = 9.81;      % gravity [m/s²]
    params.d_drag  = 0.08;      % linear drag coefficient [N·s/m]
                                %   (small frame, low drag — tunable)

    % Geometry (2D sim uses one axis of the frame)
    % Frame is 30 cm × 40 cm motor-to-motor.
    % For the 2D side-view sim, we model the pair of motors along the
    % LONGER axis (40 cm), giving L = half-span = 0.20 m.
    params.L       = 0.20;      % half-distance between motor pair [m]

    % Moment of inertia about pitch axis (2D rotation axis)
    % Estimated for a 686 g drone with distributed mass:
    %   4 motors (9.1 g each) at ±0.20 m  → 4×0.0091×0.20² = 0.00146
    %   battery (~200 g) near CoM          → ~0.0005
    %   frame + electronics (~250 g)       → ~0.0015
    %   Total ≈ 0.0035 kg·m²
    params.I       = 0.0035;    % pitch moment of inertia [kg·m²]

    %% Motor constraints (per motor)
    % iFlight XING2 1404 4600KV on 4S with 4×3 prop
    %   Max thrust:    574 g ≈ 5.63 N  (from thrust test data)
    %   Peak current:  17.99 A
    %   Max power:     287.8 W
    %   T/W ratio:     4×5.63 / (0.686×9.81) = 3.35  ✓
    params.F_min     = 0.0;     % minimum thrust per motor [N]
    params.F_max     = 5.63;    % maximum thrust per motor [N]  (574 g)
    params.max_omega = 8.0;     % yaw/pitch rate limit [rad/s]
    params.max_tilt  = deg2rad(55);  % pitch angle limit [rad]
                                %   (aggressive, T/W supports it)

    % Derived quantities
    params.F_hover   = params.m * params.g / 2;  % per-motor hover thrust
    %   = 0.686×9.81/2 = 3.37 N  (60% of max → healthy headroom)

    % Derived dynamics limits (for reference):
    %   Max total thrust:   2×5.63 = 11.26 N  (2D: 2 motors)
    %   Max torque:         5.63 × 0.20 = 1.126 N·m
    %   Max angular accel:  1.126 / 0.0035 = 322 rad/s²
    %   Max vertical accel: 11.26/0.686 − 9.81 = 6.6 m/s²
    %   Max horizontal accel (at 55° tilt): ~5.4 m/s²

    %% Safety
    params.drone_radius = 0.25; % safety margin from walls [m]
                                %   (diagonal ~25 cm for 30×40 frame)

    %% MPC controller
    mpc.N       = 25;           % prediction horizon steps
    mpc.dt_ctrl = 0.04;         % control period [s]  (25 Hz)
    mpc.T_pred  = mpc.N * mpc.dt_ctrl;  % prediction horizon [s] = 1.0 s

    % Cost weights                x     y     th    vx    vy    omega
    mpc.Q  = diag([50,   50,   10,   5,    5,    0.5]);    % stage
    mpc.Qf = diag([100,  100,  20,   10,   10,   1  ]);    % terminal
    %                              F1    F2
    mpc.R  = diag([0.1,  0.1]);          % input cost
    mpc.dR = diag([4.0,  4.0 ]);         % input rate (smoothness)
    %   Increased dR because lighter drone is more sensitive to
    %   thrust transients — smoother commands = less oscillation.

    %% Path planner
    plan.T_replan  = 1.0;       % replan interval [s]
    plan.T_horizon = 2.0;       % planning horizon [s]
    plan.dt_plan   = 0.05;      % planner timestep [s]
    plan.N_steps   = round(plan.T_horizon / plan.dt_plan);

    %% Simulation
    sim_cfg.T_final = 30.0;     % total simulation time [s]
    sim_cfg.dt      = 0.001;    % plant integration timestep [s]
end
