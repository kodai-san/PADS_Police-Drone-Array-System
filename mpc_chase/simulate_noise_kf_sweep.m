function simulate_noise_kf_sweep()
% Noise & Kalman Filter parameter sweep for MPC drone chase.
%
% Sweeps over:
%   - sigma_drone  : drone GPS position noise std (m)
%   - sigma_target : target tracker position noise std (m)
%   - q_intensity  : KF process noise intensity (higher = trust measurements more)
%
% For each combination, runs a headless Westminster chase simulation with:
%   - Gaussian measurement noise on drone and target positions
%   - Separate Kalman Filters for drone self-localisation and target tracking
%   - MPC controller uses KF-filtered states
%
% Outputs:
%   - Heatmap plots of key metrics vs noise levels for each q
%   - Line plots of metrics vs q for fixed noise
%   - Console summary table
%
% Requires: Optimization Toolbox (quadprog)
%           westminster_map.mat in current directory

% Ensure map_extract subfolder is on the path
addpath(fullfile(fileparts(mfilename('fullpath')), 'map_extract'));

%% =========================
%  Sweep grid
% =========================
sigma_drone_vals  = [0, 0.25, 0.5, 1.0, 2.0];
sigma_target_vals = [0, 0.5, 1.0, 2.0, 4.0];
q_intensity_vals  = [0.1, 0.2, 0.3, 0.5, 1.0];

nSD = numel(sigma_drone_vals);
nST = numel(sigma_target_vals);
nQ  = numel(q_intensity_vals);
nTotal = nSD * nST * nQ;

%% =========================
%  Load map (once)
% =========================
fprintf('Loading Westminster map...\n');
mapData    = load_westminster_map('westminster_map.mat');
env        = makeWestminsterEnvironment(mapData);
targetPath = mapData.target_path;

%% =========================
%  Common parameters
% =========================
[P, W] = defaultParameters();
envInf = inflateBuildings(env, P.obstacleInflation);

%% =========================
%  Run sweep
% =========================
results = struct();
runIdx = 0;

fprintf('\n====================================================\n');
fprintf('  NOISE + KF PARAMETER SWEEP  (%d runs)\n', nTotal);
fprintf('====================================================\n\n');

for iQ = 1:nQ
    for iSD = 1:nSD
        for iST = 1:nST
            runIdx = runIdx + 1;
            sd = sigma_drone_vals(iSD);
            st = sigma_target_vals(iST);
            qi = q_intensity_vals(iQ);

            fprintf('Run %3d/%d: sigma_d=%.2f, sigma_t=%.2f, q=%.1f ... ', ...
                runIdx, nTotal, sd, st, qi);

            m = runOneChase(sd, st, qi, mapData, env, envInf, targetPath, P, W);

            results(iSD, iST, iQ).sigma_drone  = sd;
            results(iSD, iST, iQ).sigma_target = st;
            results(iSD, iST, iQ).q_intensity  = qi;
            results(iSD, iST, iQ).metrics      = m;

            fprintf('dist_err=%.2fm, E=%.0fJ, osc=%.4f\n', ...
                m.dist_error_rms, m.total_energy_J, m.oscillation_score);
        end
    end
end

fprintf('\nAll %d runs complete.\n\n', nTotal);

%% =========================
%  Print summary table
% =========================
printSummaryTable(results, sigma_drone_vals, sigma_target_vals, q_intensity_vals);

%% =========================
%  Generate plots
% =========================
plotSweepResults(results, sigma_drone_vals, sigma_target_vals, q_intensity_vals);

fprintf('Done. Figures are open.\n');
end

%% ========================================================================
%  DEFAULT PARAMETERS
%% ========================================================================
function [P, W] = defaultParameters()
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
P.vt        = 5;
P.R_des     = 8.0;
P.usePredictionWhenNoLoS = true;
P.obstacleInflation = 1.0;
P.safetyCandidates  = 24;
P.safeStepLookahead = 1;
P.emergencyBrake    = true;
P.losBuildingRange  = 80;
P.saveVideo = false;

W.w_pos = 20.0;
W.w_vel = 2.0;
W.w_u   = 0.5;
W.w_du  = 3.0;
end

%% ========================================================================
%  SINGLE CHASE RUN (headless, with noise + KF)
%% ========================================================================
function metrics = runOneChase(sigma_drone, sigma_target, q_intensity, ...
                               mapData, env, envInf, targetPath, P, W)

% --- Simulation time ---
pathLen = sum(sqrt(sum(diff(targetPath).^2, 2)));
P.Tsim  = pathLen / P.vt + 10;
K = floor(P.Tsim / P.dt) + 1;
t = (0:K-1)' * P.dt;

% --- Init true states ---
initVelDir = targetPath(2,:)' - targetPath(1,:)';
initVelDir = initVelDir / (norm(initVelDir) + 1e-6);
xt = [targetPath(1,:)'; P.vt * initVelDir];
wpIdx = 2;

initDir = initVelDir;
xd = [targetPath(1,:)' - P.R_des * initDir; 0; 0];
phi = 0; theta = 0;

% --- Dynamics matrices ---
nx = 4; nu = 2;
A = [1 0 P.dt 0; 0 1 0 P.dt; 0 0 1 0; 0 0 0 1];
B = [0.5*P.dt^2 0; 0 0.5*P.dt^2; P.dt 0; 0 P.dt];
[Phi_mpc, Gamma_mpc] = buildPredictionMatrices(A, B, P.N);
Du = diffMatrix2D(P.N);

% --- Kalman Filter setup ---
H = [eye(2), zeros(2)];  % measure position only

% Process noise covariance (discrete, from continuous q_intensity)
% Continuous: q_c * I.  Discrete approximation for const-velocity model:
Q_kf = q_intensity * [P.dt^3/3*eye(2), P.dt^2/2*eye(2);
                       P.dt^2/2*eye(2), P.dt*eye(2)];

R_drone  = sigma_drone^2  * eye(2) + 1e-12*eye(2);  % measurement noise cov
R_target = sigma_target^2 * eye(2) + 1e-12*eye(2);

% Drone KF state
xd_kf = xd;                  % KF estimate of drone state
Pd_kf = 0.1 * eye(nx);       % KF covariance for drone

% Target KF state
xt_kf = xt;                  % KF estimate of target state
Pt_kf = 1.0 * eye(nx);       % KF covariance for target

% --- Logs ---
log.xd  = zeros(K,4);   % true drone state
log.xt  = zeros(K,4);   % true target state
log.xd_kf = zeros(K,4); % KF drone estimate
log.xt_kf = zeros(K,4); % KF target estimate
log.phi = zeros(K,1);
log.theta = zeros(K,1);
log.los = false(K,1);
log.dist = zeros(K,1);
log.vd  = zeros(K,1);
log.axy = zeros(K,2);
log.safeUsed  = false(K,1);
log.thrust_mag   = zeros(K,1);
log.power_mech   = zeros(K,1);
log.energy_cum   = zeros(K,1);
log.jerk_xy      = zeros(K,2);
log.heading_rate = zeros(K,1);
log.dist_error   = zeros(K,1);
log.drone_pos_err  = zeros(K,1);  % KF estimation error (drone)
log.target_pos_err = zeros(K,1);  % KF estimation error (target)

u_prev = [0;0];
energy_accum = 0;

% Pre-filter nearby buildings
nearbyBldgUpdate = 0;
nearbyBldg = env.buildings;

% --- Main loop ---
for k = 1:K
    % === LOG ===
    log.xd(k,:) = xd';
    log.xt(k,:) = xt';
    log.xd_kf(k,:) = xd_kf';
    log.xt_kf(k,:) = xt_kf';
    log.phi(k) = phi;
    log.theta(k) = theta;
    log.dist(k) = norm(xt(1:2) - xd(1:2));
    log.vd(k)   = norm(xd(3:4));
    log.axy(k,:) = u_prev';
    log.dist_error(k) = abs(log.dist(k) - P.R_des);
    log.drone_pos_err(k)  = norm(xd_kf(1:2) - xd(1:2));
    log.target_pos_err(k) = norm(xt_kf(1:2) - xt(1:2));

    % === ENERGY ===
    a_horiz = u_prev;
    thrust = P.mass * sqrt(a_horiz(1)^2 + a_horiz(2)^2 + P.g^2);
    log.thrust_mag(k) = thrust;
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
    if k > 1 && log.vd(k) > 0.3 && log.vd(k-1) > 0.3
        h_now  = atan2(xd(4), xd(3));
        h_prev = atan2(log.xd(k-1,4), log.xd(k-1,3));
        dh = atan2(sin(h_now - h_prev), cos(h_now - h_prev));
        log.heading_rate(k) = dh / P.dt;
    end

    % === DRONE KF: PREDICT ===
    % Predict using the control input we applied last step
    xd_kf_pred = A * xd_kf + B * u_prev;
    Pd_kf_pred = A * Pd_kf * A' + Q_kf;

    % === DRONE KF: UPDATE (noisy GPS measurement) ===
    z_drone = xd(1:2) + sigma_drone * randn(2,1);  % noisy position
    y_drone = z_drone - H * xd_kf_pred;             % innovation
    S_drone = H * Pd_kf_pred * H' + R_drone;
    K_drone = Pd_kf_pred * H' / S_drone;
    xd_kf   = xd_kf_pred + K_drone * y_drone;
    Pd_kf   = (eye(nx) - K_drone * H) * Pd_kf_pred;

    % === NEARBY BUILDINGS ===
    if t(k) - nearbyBldgUpdate > 1.0
        nearbyBldg = filterNearbyBuildings(xd_kf(1:2), env.buildings, P.losBuildingRange);
        nearbyBldgUpdate = t(k);
    end

    % === LoS (use true positions for physics, but KF could affect check) ===
    los = hasLineOfSight(xd(1:2), xt(1:2), nearbyBldg);
    log.los(k) = los;

    % === TARGET KF: PREDICT (constant-velocity model, no control) ===
    xt_kf_pred = A * xt_kf;  % no B*u for target
    Pt_kf_pred = A * Pt_kf * A' + Q_kf;

    if los
        % === TARGET KF: UPDATE (noisy tracker measurement) ===
        z_target = xt(1:2) + sigma_target * randn(2,1);
        y_target = z_target - H * xt_kf_pred;
        S_target = H * Pt_kf_pred * H' + R_target;
        K_target = Pt_kf_pred * H' / S_target;
        xt_kf    = xt_kf_pred + K_target * y_target;
        Pt_kf    = (eye(nx) - K_target * H) * Pt_kf_pred;
    else
        % No measurement: predict only
        xt_kf = xt_kf_pred;
        Pt_kf = Pt_kf_pred;
    end

    % === REFERENCE (use KF-filtered target estimate) ===
    pref = zeros(2, P.N+1);
    vref = zeros(2, P.N+1);
    for i = 1:P.N+1
        dt_fwd = (i-1) * P.dt;
        futurePos = xt_kf(1:2) + xt_kf(3:4) * dt_fwd;
        futureVel = xt_kf(3:4);
        dir = futureVel / (norm(futureVel) + 1e-6);
        pref(:,i) = futurePos - P.R_des * dir;
        vref(:,i) = futureVel;
    end

    % === SOLVE MPC (using KF-filtered drone state) ===
    U = solveMPC_QP(xd_kf, u_prev, pref, vref, Phi_mpc, Gamma_mpc, Du, P, W);

    if any(isnan(U))
        a_des = [0;0];
    else
        a_des = U(1:2);
    end

    % === TILT DYNAMICS ===
    theta_cmd = clampScalar(atan2(a_des(1), P.g), P.tiltMax);
    phi_cmd   = clampScalar(-atan2(a_des(2), P.g), P.tiltMax);
    [phi, theta] = tiltStep(phi, theta, phi_cmd, theta_cmd, P);

    accel_x = P.g * tan(theta);
    accel_y = -P.g * tan(phi);
    a_act = clamp2([accel_x; accel_y], P.amax);

    % === SAFETY FILTER (uses true drone state for safety) ===
    [a_safe, used] = safetyFilterAccel(xd, a_act, A, B, envInf, P);
    log.safeUsed(k) = used;

    % === PROPAGATE TRUE DRONE ===
    xd = A * xd + B * a_safe;
    xd(3:4) = clamp2(xd(3:4), P.vmax);
    if pointInAnyBuilding(xd(1:2), envInf.buildings)
        xd(3:4) = [0;0];
    end
    u_prev = a_safe;

    % === PROPAGATE TRUE TARGET ===
    [xt, wpIdx] = targetStepWaypoint(xt, wpIdx, targetPath, P.vt, P.dt);
    if wpIdx > size(targetPath, 1) && norm(xt(3:4)) < 0.01
        K = k;
        break;
    end
end

% --- Truncate & compute metrics ---
t = t(1:K);
fields = fieldnames(log);
for f = 1:numel(fields)
    fn = fields{f};
    log.(fn) = log.(fn)(1:K, :);
end

metrics = computeSweepMetrics(t, log, P);
end

%% ========================================================================
%  METRICS COMPUTATION
%% ========================================================================
function metrics = computeSweepMetrics(t, log, P)
dt = P.dt;

% Distance tracking
dist_err = log.dist - P.R_des;
metrics.dist_error_rms  = rms(dist_err);
metrics.dist_error_mean = mean(abs(dist_err));
metrics.dist_error_peak = max(abs(dist_err));

% Energy
metrics.total_energy_J  = log.energy_cum(end);
metrics.total_energy_Wh = log.energy_cum(end) / 3600;
metrics.mean_power_W    = mean(log.power_mech);

hover_thrust = P.mass * P.g;
hover_power  = hover_thrust^(1.5) / sqrt(2 * P.rho * P.A_disc);
metrics.hover_energy_J  = hover_power * t(end);

% Jerk
jerk_mag = sqrt(log.jerk_xy(:,1).^2 + log.jerk_xy(:,2).^2);
metrics.jerk_rms  = rms(jerk_mag);
metrics.jerk_peak = max(jerk_mag);

% LoS & safety
metrics.los_fraction    = mean(log.los);
metrics.safety_fraction = mean(log.safeUsed);

% KF estimation errors
metrics.drone_pos_err_rms  = rms(log.drone_pos_err);
metrics.target_pos_err_rms = rms(log.target_pos_err);

% Oscillation score
jerk_norm = metrics.jerk_rms / (P.amax / dt);
dist_norm = metrics.dist_error_rms / P.R_des;
metrics.oscillation_score = 0.5 * jerk_norm + 0.5 * dist_norm;
end

%% ========================================================================
%  FILTER NEARBY BUILDINGS (local copy for headless runs)
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
%  PRINT SUMMARY TABLE
%% ========================================================================
function printSummaryTable(results, sd_vals, st_vals, q_vals)
fprintf('\n');
fprintf('========================================================================\n');
fprintf('  NOISE + KF SWEEP — SUMMARY TABLE\n');
fprintf('========================================================================\n');
fprintf('  %-6s  %-6s  %-6s | %-8s  %-8s  %-8s  %-8s  %-8s\n', ...
    'sig_d', 'sig_t', 'q', 'distRMS', 'energy', 'jerkRMS', 'oscScore', 'tgtErr');
fprintf('  %s\n', repmat('-', 1, 72));

for iQ = 1:numel(q_vals)
    for iSD = 1:numel(sd_vals)
        for iST = 1:numel(st_vals)
            r = results(iSD, iST, iQ);
            m = r.metrics;
            fprintf('  %6.2f  %6.2f  %6.1f | %7.3f  %8.0f  %7.2f  %8.4f  %7.3f\n', ...
                r.sigma_drone, r.sigma_target, r.q_intensity, ...
                m.dist_error_rms, m.total_energy_J, m.jerk_rms, ...
                m.oscillation_score, m.target_pos_err_rms);
        end
    end
    fprintf('  %s\n', repmat('-', 1, 72));
end
fprintf('\n');
end

%% ========================================================================
%  PLOT SWEEP RESULTS
%% ========================================================================
function plotSweepResults(results, sd_vals, st_vals, q_vals)
nSD = numel(sd_vals);
nST = numel(st_vals);
nQ  = numel(q_vals);

% Extract metric arrays
distRMS  = zeros(nSD, nST, nQ);
energy   = zeros(nSD, nST, nQ);
jerkRMS  = zeros(nSD, nST, nQ);
oscScore = zeros(nSD, nST, nQ);
dronePE  = zeros(nSD, nST, nQ);
targetPE = zeros(nSD, nST, nQ);

for iQ = 1:nQ
    for iSD = 1:nSD
        for iST = 1:nST
            m = results(iSD, iST, iQ).metrics;
            distRMS(iSD, iST, iQ)  = m.dist_error_rms;
            energy(iSD, iST, iQ)   = m.total_energy_J;
            jerkRMS(iSD, iST, iQ)  = m.jerk_rms;
            oscScore(iSD, iST, iQ) = m.oscillation_score;
            dronePE(iSD, iST, iQ)  = m.drone_pos_err_rms;
            targetPE(iSD, iST, iQ) = m.target_pos_err_rms;
        end
    end
end

% ---- Figure 1: Heatmaps of dist error RMS for each q ----
figure('Color','w','Name','Distance Error vs Noise (per q)','Position',[50 50 1200 350]);
for iQ = 1:nQ
    subplot(1, nQ, iQ);
    imagesc(st_vals, sd_vals, distRMS(:,:,iQ));
    colorbar; axis xy;
    xlabel('\sigma_{target} (m)'); ylabel('\sigma_{drone} (m)');
    title(sprintf('Dist Err RMS — q=%.1f', q_vals(iQ)));
    colormap(gca, 'parula');
end
sgtitle('Distance Error RMS (m) vs Noise Levels');

% ---- Figure 2: Heatmaps of oscillation score for each q ----
figure('Color','w','Name','Oscillation Score vs Noise (per q)','Position',[50 420 1200 350]);
for iQ = 1:nQ
    subplot(1, nQ, iQ);
    imagesc(st_vals, sd_vals, oscScore(:,:,iQ));
    colorbar; axis xy;
    xlabel('\sigma_{target} (m)'); ylabel('\sigma_{drone} (m)');
    title(sprintf('Osc Score — q=%.1f', q_vals(iQ)));
    colormap(gca, 'hot');
end
sgtitle('Oscillation Score vs Noise Levels');

% ---- Figure 3: Heatmaps of energy for each q ----
figure('Color','w','Name','Energy vs Noise (per q)','Position',[50 790 1200 350]);
for iQ = 1:nQ
    subplot(1, nQ, iQ);
    imagesc(st_vals, sd_vals, energy(:,:,iQ));
    colorbar; axis xy;
    xlabel('\sigma_{target} (m)'); ylabel('\sigma_{drone} (m)');
    title(sprintf('Energy — q=%.1f', q_vals(iQ)));
    colormap(gca, 'parula');
end
sgtitle('Total Energy (J) vs Noise Levels');

% ---- Figure 4: Metrics vs q (fixed noise at default: sig_d=0.5, sig_t=1.0) ----
% Find indices closest to default noise
[~, iSD_def] = min(abs(sd_vals - 0.5));
[~, iST_def] = min(abs(st_vals - 1.0));

figure('Color','w','Name','Metrics vs KF Process Noise q','Position',[50 50 900 700]);

subplot(2,2,1);
plot(q_vals, squeeze(distRMS(iSD_def, iST_def, :)), 'o-', 'LineWidth', 1.8, 'MarkerSize', 7);
grid on; xlabel('q (process noise intensity)'); ylabel('Dist Error RMS (m)');
title(sprintf('Distance Error (\\sigma_d=%.1f, \\sigma_t=%.1f)', sd_vals(iSD_def), st_vals(iST_def)));
set(gca, 'XScale', 'log');

subplot(2,2,2);
plot(q_vals, squeeze(oscScore(iSD_def, iST_def, :)), 's-', 'LineWidth', 1.8, 'MarkerSize', 7, 'Color', [0.85 0.33 0.1]);
grid on; xlabel('q (process noise intensity)'); ylabel('Oscillation Score');
title('Oscillation Score');
set(gca, 'XScale', 'log');

subplot(2,2,3);
plot(q_vals, squeeze(energy(iSD_def, iST_def, :)), 'd-', 'LineWidth', 1.8, 'MarkerSize', 7, 'Color', [0.47 0.67 0.19]);
grid on; xlabel('q (process noise intensity)'); ylabel('Energy (J)');
title('Total Energy');
set(gca, 'XScale', 'log');

subplot(2,2,4);
plot(q_vals, squeeze(jerkRMS(iSD_def, iST_def, :)), '^-', 'LineWidth', 1.8, 'MarkerSize', 7, 'Color', [0.49 0.18 0.56]);
grid on; xlabel('q (process noise intensity)'); ylabel('Jerk RMS (m/s^3)');
title('Jerk RMS');
set(gca, 'XScale', 'log');

sgtitle(sprintf('Effect of KF Tuning (\\sigma_d=%.1f m, \\sigma_t=%.1f m)', ...
    sd_vals(iSD_def), st_vals(iST_def)));

% ---- Figure 5: KF estimation error vs noise ----
figure('Color','w','Name','KF Estimation Errors','Position',[960 50 700 500]);

subplot(2,1,1);
hold on;
for iQ = 1:nQ
    plot(sd_vals, dronePE(:, iST_def, iQ), 'o-', 'LineWidth', 1.5, 'DisplayName', sprintf('q=%.1f', q_vals(iQ)));
end
grid on; xlabel('\sigma_{drone} (m)'); ylabel('Drone Pos Err RMS (m)');
title('Drone KF Estimation Error'); legend('Location','northwest');

subplot(2,1,2);
hold on;
for iQ = 1:nQ
    plot(st_vals, targetPE(iSD_def, :, iQ), 's-', 'LineWidth', 1.5, 'DisplayName', sprintf('q=%.1f', q_vals(iQ)));
end
grid on; xlabel('\sigma_{target} (m)'); ylabel('Target Pos Err RMS (m)');
title('Target KF Estimation Error'); legend('Location','northwest');

sgtitle('Kalman Filter Estimation Accuracy');

% ---- Figure 6: Multi-line comparison for different noise combos ----
figure('Color','w','Name','Multi-noise Comparison','Position',[960 570 700 500]);

combos = [1 1; 3 3; 5 5];  % (iSD, iST) pairs: low, mid, high noise
labels = {};
hold on;
colors = lines(size(combos,1));
for c = 1:size(combos,1)
    iSD_c = combos(c,1); iST_c = combos(c,2);
    vals = squeeze(distRMS(iSD_c, iST_c, :));
    plot(q_vals, vals, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'Color', colors(c,:));
    labels{c} = sprintf('\\sigma_d=%.2f, \\sigma_t=%.1f', sd_vals(iSD_c), st_vals(iST_c));
end
grid on; set(gca, 'XScale', 'log');
xlabel('q (process noise intensity)'); ylabel('Distance Error RMS (m)');
title('Distance Error vs KF Tuning at Different Noise Levels');
legend(labels, 'Location', 'best');
end
