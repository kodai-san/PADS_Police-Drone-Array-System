function simulate_westminster_chase()
% MPC drone chase simulation on real Westminster (London) map data.
%
% The target walks along a pre-computed route through Westminster streets.
% The drone tracks it using MPC with:
%   - Real building footprints for LoS occlusion
%   - Building collision avoidance (safety filter)
%   - Roll/pitch dynamics
%   - Energy and oscillation metrics
%
% Prerequisites:
%   1. Run: python extract_westminster_map.py   (in map_extract/)
%   2. Copy westminster_map.mat to this directory
%   3. Run: simulate_westminster_chase
%
% Requires: Optimization Toolbox (quadprog)

% Ensure map_extract subfolder is on the path
addpath(fullfile(fileparts(mfilename('fullpath')), 'map_extract'));

%% =========================
%  Parameters
% =========================
P.dt        = 1/20;
P.N         = 30;       % longer horizon helps in city streets

P.bladeDist = 0.40;
P.vmax      = 15.0;     % m/s
P.amax      = 5.0;      % m/s^2
P.tiltMaxDeg= 30.0;
P.tiltMax   = deg2rad(P.tiltMaxDeg);
P.g         = 9.81;
P.tiltTau   = 0.20;
P.tiltRateMax = deg2rad(200);
P.mass      = 2.5;
P.rho       = 1.225;
P.A_disc    = 0.04;

% Target
P.vt        = 5;      % m/s (~typical walking speed)

% Tracking
P.R_des     = 8.0;      % following distance (wider for urban canyons)
P.usePredictionWhenNoLoS = true;

% MPC weights
W.w_pos = 15.0;
W.w_vel = 0.1;
W.w_u   = 2.93;
W.w_du  = 3.13;

% Safety
P.obstacleInflation = 1.0;  % larger margin for real buildings
P.safetyCandidates  = 24;   % more candidates for complex geometry
P.safeStepLookahead = 1;
P.emergencyBrake    = true;

% LoS checking: limit to nearby buildings for speed
P.losBuildingRange = 80;  % only check buildings within this range (m)

% Video
P.saveVideo = true;
P.videoFile = 'westminster_chase.mp4';
P.videoFPS  = 20;

%% =========================
%  Load map
% =========================
fprintf('Loading Westminster map...\n');
mapData = load_westminster_map('westminster_map.mat');

% Build environment struct
env = makeWestminsterEnvironment(mapData);
envInf = inflateBuildings(env, P.obstacleInflation);

targetPath = mapData.target_path;  % [P x 2]

% Compute simulation time from path length
pathLen = sum(sqrt(sum(diff(targetPath).^2, 2)));
P.Tsim = pathLen / P.vt + 10;  % add buffer
K = floor(P.Tsim / P.dt) + 1;
t = (0:K-1)' * P.dt;
fprintf('Path length: %.0f m, Sim time: %.1f s, Steps: %d\n', pathLen, P.Tsim, K);

%% =========================
%  Init
% =========================
% Target starts at first path point, velocity along path direction
initVelDir = targetPath(2,:)' - targetPath(1,:)';
initVelDir = initVelDir / (norm(initVelDir) + 1e-6);
xt = [targetPath(1,:)'; P.vt * initVelDir];
wpIdx = 2;  % heading toward second point

% Drone starts behind target along the initial path direction
initDir = targetPath(2,:)' - targetPath(1,:)';
initDir = initDir / (norm(initDir) + 1e-6);
xd = [targetPath(1,:)' - P.R_des * initDir; 0; 0];

phi = 0; theta = 0;
xthat = xt; xthat_last = xt; lastMeasTime = 0;

% MPC model
A = [1 0 P.dt 0; 0 1 0 P.dt; 0 0 1 0; 0 0 0 1];
B = [0.5*P.dt^2 0; 0 0.5*P.dt^2; P.dt 0; 0 P.dt];
[Phi, Gamma] = buildPredictionMatrices(A, B, P.N);
Du = diffMatrix2D(P.N);

% Logs
log.xd  = zeros(K,4);
log.xt  = zeros(K,4);
log.phi = zeros(K,1);
log.theta = zeros(K,1);
log.los = false(K,1);
log.dist= zeros(K,1);
log.vd  = zeros(K,1);
log.axy = zeros(K,2);
log.safeUsed = false(K,1);
log.thrust_mag   = zeros(K,1);
log.power_mech   = zeros(K,1);
log.energy_cum   = zeros(K,1);
log.jerk_xy      = zeros(K,2);
log.heading_rate = zeros(K,1);
log.dist_error   = zeros(K,1);

%% =========================
%  Figure + video
% =========================
fig = figure('Color','w','Name','Westminster Chase','Position',[50 50 1000 800]);
hAx = axes(fig); hold(hAx,'on'); axis(hAx,'equal'); grid(hAx,'on');
xlabel(hAx,'East (m)'); ylabel(hAx,'North (m)');
title(hAx, 'MPC Drone Chase — Westminster, London');

drawWestminsterEnvironment(hAx, mapData, targetPath);

hTarget  = plot(hAx, xt(1), xt(2), 'r.', 'MarkerSize', 18);
hDrone   = patch(hAx, nan, nan, 'g', 'FaceAlpha', 0.4, 'EdgeColor','g', 'LineWidth', 1.5);
hLine    = plot(hAx, [xd(1) xt(1)], [xd(2) xt(2)], '-', 'Color', [0.2 0.2 0.2 0.5], 'LineWidth', 0.8);
hMpcPath = plot(hAx, nan, nan, 'c-', 'LineWidth', 1.8);
hRefPath = plot(hAx, nan, nan, 'm--', 'LineWidth', 1.0);
hText    = text(hAx, 0, 0, '', 'FontSize', 9, 'Color', 'k', 'FontName', 'FixedWidth');

% Follow camera: set initial view centered on drone
viewW = 80;  % view window half-width (m)
xlim(hAx, [xd(1)-viewW, xd(1)+viewW]);
ylim(hAx, [xd(2)-viewW, xd(2)+viewW]);

if P.saveVideo
    vw = VideoWriter(P.videoFile, 'MPEG-4');
    vw.FrameRate = P.videoFPS;
    open(vw);
else
    vw = [];
end

%% =========================
%  Main Loop
% =========================
u_prev = [0;0];
energy_accum = 0;
nx = 4; nu = 2;

% Pre-filter nearby buildings for LoS (recompute occasionally)
nearbyBldgUpdate = 0;
nearbyBldg = env.buildings;

for k = 1:K
    % --- Log ---
    log.xd(k,:) = xd';
    log.xt(k,:) = xt';
    log.phi(k) = phi;
    log.theta(k) = theta;
    log.dist(k) = norm(xt(1:2)-xd(1:2));
    log.vd(k) = norm(xd(3:4));
    log.axy(k,:) = u_prev';
    log.dist_error(k) = abs(log.dist(k) - P.R_des);

    % --- Energy ---
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

    % --- Update nearby building list periodically (every 1s) ---
    if t(k) - nearbyBldgUpdate > 1.0
        nearbyBldg = filterNearbyBuildings(xd(1:2), env.buildings, P.losBuildingRange);
        nearbyBldgUpdate = t(k);
    end

    % --- LoS (use nearby buildings only for speed) ---
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

    % --- Reference: constant-velocity prediction (no path foreknowledge) ---
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

    % --- Solve MPC ---
    U = solveMPC_QP(xd, u_prev, pref, vref, Phi, Gamma, Du, P, W);

    if any(isnan(U))
        a_des = [0;0];
        mpc_px = nan; mpc_py = nan;
    else
        a_des = U(1:2);
        X_pred = Phi * xd + Gamma * U;
        mpc_px = zeros(1, P.N+1);
        mpc_py = zeros(1, P.N+1);
        for i = 1:P.N+1
            idx = (i-1)*nx + (1:nx);
            mpc_px(i) = X_pred(idx(1));
            mpc_py(i) = X_pred(idx(2));
        end
    end

    % --- Update overlays ---
    set(hMpcPath, 'XData', mpc_px, 'YData', mpc_py);
    set(hRefPath, 'XData', pref(1,:), 'YData', pref(2,:));

    % --- Tilt dynamics ---
    theta_cmd = clampScalar(atan2(a_des(1), P.g), P.tiltMax);
    phi_cmd   = clampScalar(-atan2(a_des(2), P.g), P.tiltMax);
    [phi, theta] = tiltStep(phi, theta, phi_cmd, theta_cmd, P);

    accel_x = P.g * tan(theta);
    accel_y = -P.g * tan(phi);
    a_act = clamp2([accel_x; accel_y], P.amax);

    % --- Safety filter ---
    [a_safe, used] = safetyFilterAccel(xd, a_act, A, B, envInf, P);
    log.safeUsed(k) = used;

    % --- Propagate drone ---
    xd = A*xd + B*a_safe;
    xd(3:4) = clamp2(xd(3:4), P.vmax);

    if pointInAnyBuilding(xd(1:2), envInf.buildings)
        xd(3:4) = [0;0];
    end

    u_prev = a_safe;

    % --- Propagate target along path ---
    [xt, wpIdx] = targetStepWaypoint(xt, wpIdx, targetPath, P.vt, P.dt);

    % Stop if target finished path
    if wpIdx > size(targetPath, 1) && norm(xt(3:4)) < 0.01
        fprintf('Target reached end of path at t=%.1fs\n', t(k));
        K = k;  % truncate
        break;
    end

    % --- Draw ---
    set(hTarget, 'XData', xt(1), 'YData', xt(2));
    [sx, sy] = squarePatch(xd(1), xd(2), P.bladeDist);
    set(hDrone, 'XData', sx, 'YData', sy);
    set(hLine, 'XData', [xd(1) xt(1)], 'YData', [xd(2) xt(2)]);

    % Follow camera
    camX = (xd(1) + xt(1)) / 2;
    camY = (xd(2) + xt(2)) / 2;
    xlim(hAx, [camX-viewW, camX+viewW]);
    ylim(hAx, [camY-viewW, camY+viewW]);

    status = sprintf('t=%6.2fs | LoS=%d | dist=%5.1fm | v=%4.1fm/s | E=%6.0fJ | saf=%d', ...
        t(k), los, log.dist(k), log.vd(k), energy_accum, used);
    xl = xlim(hAx); yl = ylim(hAx);
    set(hText, 'Position', [xl(1)+2, yl(2)-3, 0], 'String', status);

    drawnow limitrate;

    if P.saveVideo
        frame = getframe(fig);
        writeVideo(vw, frame);
    end
end

if P.saveVideo
    close(vw);
    fprintf('Saved video to: %s\n', P.videoFile);
end

%% =========================
%  Truncate logs & metrics
% =========================
t = t(1:K);
fields = fieldnames(log);
for f = 1:numel(fields)
    fn = fields{f};
    log.(fn) = log.(fn)(1:K, :);
end

metrics = computeMetrics(t, log, P);
printMetrics(metrics);

%% =========================
%  Post-sim plots
% =========================
figure('Color','w','Name','Westminster Chase — Metrics');

subplot(4,1,1);
plot(t, log.dist, 'LineWidth', 1.5); hold on;
yline(P.R_des,'--'); grid on;
ylabel('dist (m)'); title('Distance to target');

subplot(4,1,2);
plot(t, log.vd, 'LineWidth', 1.5); grid on;
ylabel('speed (m/s)');

subplot(4,1,3);
plot(t, log.energy_cum, 'LineWidth', 1.5, 'Color', [0.85 0.33 0.1]);
grid on; ylabel('Energy (J)'); title('Cumulative Energy');

subplot(4,1,4);
stairs(t, log.los, 'LineWidth', 1.2); hold on;
stairs(t, log.safeUsed, 'LineWidth', 1.2);
grid on; ylim([-0.1 1.1]); xlabel('time (s)');
legend('LoS','Safety filter');

end

%% ========================================================================
%  Helper: filter nearby buildings for fast LoS
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
%  Metrics (local)
%% ========================================================================
function metrics = computeMetrics(t, log, P)
dt = P.dt;
metrics.total_energy_J  = log.energy_cum(end);
metrics.total_energy_Wh = log.energy_cum(end) / 3600;
metrics.mean_power_W    = mean(log.power_mech);
metrics.peak_power_W    = max(log.power_mech);

hover_thrust = P.mass * P.g;
hover_power  = hover_thrust^(1.5) / sqrt(2 * P.rho * P.A_disc);
metrics.hover_energy_J = hover_power * t(end);

jerk_mag = sqrt(log.jerk_xy(:,1).^2 + log.jerk_xy(:,2).^2);
metrics.jerk_rms  = rms(jerk_mag);
metrics.jerk_peak = max(jerk_mag);

dist_err = log.dist - P.R_des;
metrics.dist_error_rms  = rms(dist_err);
metrics.dist_error_mean = mean(abs(dist_err));

metrics.los_fraction = mean(log.los);
metrics.safety_fraction = mean(log.safeUsed);

jerk_norm    = metrics.jerk_rms / (P.amax / dt);
dist_norm    = metrics.dist_error_rms / P.R_des;
metrics.oscillation_score = 0.5 * jerk_norm + 0.5 * dist_norm;
end

function printMetrics(m)
fprintf('\n');
fprintf('====================================================\n');
fprintf('  WESTMINSTER CHASE — MISSION METRICS\n');
fprintf('====================================================\n');
fprintf('  Total energy:      %8.1f J  (%.4f Wh)\n', m.total_energy_J, m.total_energy_Wh);
fprintf('  Hover baseline:    %8.1f J\n', m.hover_energy_J);
fprintf('  Mean power:        %8.1f W\n', m.mean_power_W);
fprintf('  Peak power:        %8.1f W\n', m.peak_power_W);
fprintf('  Jerk RMS:          %8.2f m/s^3\n', m.jerk_rms);
fprintf('  Dist error RMS:    %8.2f m\n', m.dist_error_rms);
fprintf('  Dist error mean:   %8.2f m\n', m.dist_error_mean);
fprintf('  LoS fraction:      %8.1f%%\n', m.los_fraction * 100);
fprintf('  Safety active:     %8.1f%%\n', m.safety_fraction * 100);
fprintf('  Oscillation score: %8.4f\n', m.oscillation_score);
fprintf('====================================================\n\n');
end
