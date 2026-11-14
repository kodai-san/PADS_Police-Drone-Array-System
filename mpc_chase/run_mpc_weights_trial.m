function run_mpc_weights_trial()
% Headless MPC simulation for weight optimization.
%
% Reads MPC weights from '_opt_weights.json', runs one Westminster chase
% simulation (no visualisation), and writes performance metrics to
% '_opt_metrics.json'.
%
% Called by: optimize_mpc_weights.py (Bayesian optimizer)
% Requires:  Optimization Toolbox (quadprog), westminster_map.mat

% Ensure map_extract subfolder is on the path
addpath(fullfile(fileparts(mfilename('fullpath')), 'map_extract'));

%% Read weights from JSON
weightsFile = '_opt_weights.json';
metricsFile = '_opt_metrics.json';

try
    raw = fileread(weightsFile);
    wIn = jsondecode(raw);
catch ME
    fprintf('ERROR reading weights file: %s\n', ME.message);
    writeErrorMetrics(metricsFile);
    return;
end

%% Default parameters
P.dt        = 1/20;
P.N         = 30;
P.bladeDist = 0.40;
P.vmax      = 15.0;
P.amax      = 5.0;
P.tiltMaxDeg= 30.0;
P.tiltMax   = deg2rad(P.tiltMaxDeg);
P.g         = 9.81;
P.tiltTau   = 0.20;
P.tiltRateMax = deg2rad(200);
P.mass      = 2.5;
P.rho       = 1.225;
P.A_disc    = 0.04;
P.vt        = 8;
P.R_des     = 8.0;
P.usePredictionWhenNoLoS = true;
P.obstacleInflation = 1.0;
P.safetyCandidates  = 24;
P.safeStepLookahead = 1;
P.emergencyBrake    = true;
P.losBuildingRange  = 80;
P.saveVideo = false;

% MPC weights from optimizer
W.w_pos = wIn.w_pos;
W.w_vel = wIn.w_vel;
W.w_u   = wIn.w_u;
W.w_du  = wIn.w_du;

%% Load map
mapData    = load_westminster_map('westminster_map.mat');
env        = makeWestminsterEnvironment(mapData);
envInf     = inflateBuildings(env, P.obstacleInflation);
targetPath = mapData.target_path;

%% Run headless simulation
metrics = runOneChase(mapData, env, envInf, targetPath, P, W);

%% Write metrics to JSON
outStruct = struct( ...
    'dist_error_rms',   metrics.dist_error_rms, ...
    'dist_error_mean',  metrics.dist_error_mean, ...
    'dist_error_peak',  metrics.dist_error_peak, ...
    'total_energy_J',   metrics.total_energy_J, ...
    'mean_power_W',     metrics.mean_power_W, ...
    'jerk_rms',         metrics.jerk_rms, ...
    'jerk_peak',        metrics.jerk_peak, ...
    'los_fraction',     metrics.los_fraction, ...
    'safety_fraction',  metrics.safety_fraction, ...
    'oscillation_score', metrics.oscillation_score, ...
    'success',          true ...
);

fid = fopen(metricsFile, 'w');
fprintf(fid, '%s', jsonencode(outStruct));
fclose(fid);

fprintf('Trial complete: dist_err=%.3f, E=%.0fJ, jerk=%.2f, osc=%.4f\n', ...
    metrics.dist_error_rms, metrics.total_energy_J, ...
    metrics.jerk_rms, metrics.oscillation_score);
end

%% ========================================================================
%  SINGLE CHASE RUN (headless, deterministic — no noise)
%% ========================================================================
function metrics = runOneChase(~, env, envInf, targetPath, P, W)

% Simulation time
pathLen = sum(sqrt(sum(diff(targetPath).^2, 2)));
P.Tsim  = pathLen / P.vt + 10;
K = floor(P.Tsim / P.dt) + 1;
t = (0:K-1)' * P.dt;

% Init states
initVelDir = targetPath(2,:)' - targetPath(1,:)';
initVelDir = initVelDir / (norm(initVelDir) + 1e-6);
xt = [targetPath(1,:)'; P.vt * initVelDir];
wpIdx = 2;

initDir = initVelDir;
xd = [targetPath(1,:)' - P.R_des * initDir; 0; 0];
phi = 0; theta = 0;
xthat = xt; xthat_last = xt; lastMeasTime = 0;

% Dynamics matrices
nx = 4; nu = 2;
A = [1 0 P.dt 0; 0 1 0 P.dt; 0 0 1 0; 0 0 0 1];
B = [0.5*P.dt^2 0; 0 0.5*P.dt^2; P.dt 0; 0 P.dt];
[Phi, Gamma] = buildPredictionMatrices(A, B, P.N);
Du = diffMatrix2D(P.N);

% Logs
log.dist = zeros(K,1);
log.vd   = zeros(K,1);
log.axy  = zeros(K,2);
log.los  = false(K,1);
log.safeUsed   = false(K,1);
log.power_mech = zeros(K,1);
log.energy_cum = zeros(K,1);
log.jerk_xy    = zeros(K,2);
log.dist_error = zeros(K,1);

u_prev = [0;0];
energy_accum = 0;
nearbyBldgUpdate = 0;
nearbyBldg = env.buildings;

for k = 1:K
    % Log
    log.dist(k) = norm(xt(1:2) - xd(1:2));
    log.vd(k)   = norm(xd(3:4));
    log.axy(k,:) = u_prev';
    log.dist_error(k) = abs(log.dist(k) - P.R_des);

    % Energy
    a_horiz = u_prev;
    thrust = P.mass * sqrt(a_horiz(1)^2 + a_horiz(2)^2 + P.g^2);
    speed = norm(xd(3:4));
    P_hover   = thrust^(1.5) / sqrt(2 * P.rho * P.A_disc);
    P_lateral = norm(a_horiz) * P.mass * speed;
    power_total = P_hover + P_lateral;
    log.power_mech(k) = power_total;
    energy_accum = energy_accum + power_total * P.dt;
    log.energy_cum(k) = energy_accum;

    if k > 1
        log.jerk_xy(k,:) = (log.axy(k,:) - log.axy(k-1,:)) / P.dt;
    end

    % Nearby buildings
    if t(k) - nearbyBldgUpdate > 1.0
        nearbyBldg = filterNearbyBuildings(xd(1:2), env.buildings, P.losBuildingRange);
        nearbyBldgUpdate = t(k);
    end

    % LoS
    los = hasLineOfSight(xd(1:2), xt(1:2), nearbyBldg);
    log.los(k) = los;

    if los
        xthat = xt;
        xthat_last = xthat;
        lastMeasTime = t(k);
    else
        if P.usePredictionWhenNoLoS
            dtLost = t(k) - lastMeasTime;
            xthat = xthat_last;
            xthat(1:2) = xthat(1:2) + xthat(3:4) * dtLost;
        else
            xthat = xthat_last;
        end
    end

    % Reference
    pref = zeros(2, P.N+1);
    vref = zeros(2, P.N+1);
    for i = 1:P.N+1
        dt_fwd = (i-1) * P.dt;
        futurePos = xthat(1:2) + xthat(3:4) * dt_fwd;
        futureVel = xthat(3:4);
        dir = futureVel / (norm(futureVel) + 1e-6);
        pref(:,i) = futurePos - P.R_des * dir;
        vref(:,i) = futureVel;
    end

    % MPC
    U = solveMPC_QP(xd, u_prev, pref, vref, Phi, Gamma, Du, P, W);
    if any(isnan(U))
        a_des = [0;0];
    else
        a_des = U(1:2);
    end

    % Tilt dynamics
    theta_cmd = clampScalar(atan2(a_des(1), P.g), P.tiltMax);
    phi_cmd   = clampScalar(-atan2(a_des(2), P.g), P.tiltMax);
    [phi, theta] = tiltStep(phi, theta, phi_cmd, theta_cmd, P);

    accel_x = P.g * tan(theta);
    accel_y = -P.g * tan(phi);
    a_act = clamp2([accel_x; accel_y], P.amax);

    % Safety filter
    [a_safe, used] = safetyFilterAccel(xd, a_act, A, B, envInf, P);
    log.safeUsed(k) = used;

    % Propagate drone
    xd = A*xd + B*a_safe;
    xd(3:4) = clamp2(xd(3:4), P.vmax);
    if pointInAnyBuilding(xd(1:2), envInf.buildings)
        xd(3:4) = [0;0];
    end
    u_prev = a_safe;

    % Propagate target
    [xt, wpIdx] = targetStepWaypoint(xt, wpIdx, targetPath, P.vt, P.dt);
    if wpIdx > size(targetPath, 1) && norm(xt(3:4)) < 0.01
        K = k;
        break;
    end
end

% Truncate & compute metrics
t = t(1:K);
fields = fieldnames(log);
for f = 1:numel(fields)
    fn = fields{f};
    log.(fn) = log.(fn)(1:K, :);
end

metrics = computeTrialMetrics(t, log, P);
end

%% ========================================================================
%  METRICS
%% ========================================================================
function metrics = computeTrialMetrics(t, log, P)
dt = P.dt;

dist_err = log.dist - P.R_des;
metrics.dist_error_rms  = rms(dist_err);
metrics.dist_error_mean = mean(abs(dist_err));
metrics.dist_error_peak = max(abs(dist_err));

metrics.total_energy_J  = log.energy_cum(end);
metrics.mean_power_W    = mean(log.power_mech);

hover_thrust = P.mass * P.g;
hover_power  = hover_thrust^(1.5) / sqrt(2 * P.rho * P.A_disc);
metrics.hover_energy_J  = hover_power * t(end);

jerk_mag = sqrt(log.jerk_xy(:,1).^2 + log.jerk_xy(:,2).^2);
metrics.jerk_rms  = rms(jerk_mag);
metrics.jerk_peak = max(jerk_mag);

metrics.los_fraction    = mean(log.los);
metrics.safety_fraction = mean(log.safeUsed);

jerk_norm = metrics.jerk_rms / (P.amax / dt);
dist_norm = metrics.dist_error_rms / P.R_des;
metrics.oscillation_score = 0.5 * jerk_norm + 0.5 * dist_norm;
end

%% ========================================================================
%  FILTER NEARBY BUILDINGS
%% ========================================================================
function nearby = filterNearbyBuildings(pos, buildings, range)
nearby = {};
for i = 1:numel(buildings)
    b = buildings{i};
    cx = mean(b(:,1));
    cy = mean(b(:,2));
    if norm([cx; cy] - pos(:)) < range
        nearby{end+1} = b; %#ok<AGROW>
    end
end
end

%% ========================================================================
%  ERROR FALLBACK
%% ========================================================================
function writeErrorMetrics(metricsFile)
outStruct = struct('success', false, 'error', 'Failed to read weights');
fid = fopen(metricsFile, 'w');
fprintf(fid, '%s', jsonencode(outStruct));
fclose(fid);
end
