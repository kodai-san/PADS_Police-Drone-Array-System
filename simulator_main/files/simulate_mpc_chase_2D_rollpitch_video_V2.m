function simulate_mpc_chase_2D_rollpitch_video_V2()
% 2D MPC chase simulation with:
% - roll/pitch dynamics (first-order tilt response)
% - tilt limit (±30 deg)
% - hard building obstacles (drone cannot enter)
% - LoS measurement dropouts due to buildings
% - MPC predicted path overlay on video
% - MP4 video output
% - Energy consumption metric
% - Oscillation quality metric
%
% Target is a generic moving point on road centerline (test dummy / cooperative tag).
%
% Requires: Optimization Toolbox (quadprog)

%% =========================
%  Parameters
% =========================
P.dt        = 1/20;     % 20 Hz
P.Tsim      = 25.0;     % seconds
P.N         = 40;       % horizon (1.0s)

% Geometry (visual only)
P.bladeDist = 0.40;     % meters (green square side)

% Limits
P.vmax      = 18.0;     % m/s (horizontal)
P.amax      = 5.6;      % m/s^2 (horizontal)
P.tiltMaxDeg= 20.0;     % deg
P.tiltMax   = deg2rad(P.tiltMaxDeg);
P.g         = 9.81;

% Tilt dynamics (tunable)
P.tiltTau   = 0.25;     % seconds (1st order response)
P.tiltRateMax = deg2rad(180); % rad/s clamp

% Drone mass (for energy computation)
P.mass      = 2.5;      % kg (total drone mass)

% Rotor model parameters (for power estimation)
P.rho       = 1.225;    % air density (kg/m^3)
P.A_disc    = 0.04;     % total rotor disc area (m^2)

% Target
P.vt        = 4.0;      % m/s
P.roadW     = 10.0;     % m
P.startDist = 30.0;     % m before intersection
P.startDistDrone = 5.0;
P.turnDirection = "south"; % "south" or "north"

% Tracking objective: stay behind target by R_des along target velocity
P.R_des     = 5.0;

% LoS / estimator
P.usePredictionWhenNoLoS = true;

% MPC weights (tunable)
W.w_pos = 20.0;
W.w_vel = 2.0;
W.w_u   = 0.5;
W.w_du  = 3.0;

% Obstacle safety
P.obstacleInflation = 0.5;  % meters (inflate buildings for clearance)
P.safetyCandidates  = 16;   % number of candidate directions for safety filter
P.safeStepLookahead = 1;    % apply safety based on 1-step
P.emergencyBrake    = true; % if no safe candidate, brake to stop

% Video
P.saveVideo = true;
P.videoFile = 'mpc_chase_2D.mp4';
P.videoFPS  = 20;

%% =========================
%  Init
% =========================
K = floor(P.Tsim/P.dt) + 1;
t = (0:K-1)'*P.dt;

% Target state: [px; py; vx; vy]
xt = [-P.startDist; 0; P.vt; 0];

% Drone planar state: [px; py; vx; vy]
xd = [xt(1)-P.startDistDrone; xt(2); 0; 0];

% Drone attitude states (roll phi, pitch theta)
phi = 0; theta = 0;

% Estimator memory (for target)
xthat = xt;
xthat_last = xt;
lastMeasTime = 0;

% Environment
env = makeIntersectionEnvironment(P.roadW);
envInf = inflateBuildings(env, P.obstacleInflation);

% MPC discrete planar model
A = [1 0 P.dt 0;
     0 1 0 P.dt;
     0 0 1 0;
     0 0 0 1];
B = [0.5*P.dt^2 0;
     0 0.5*P.dt^2;
     P.dt 0;
     0 P.dt];

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

% Energy & oscillation logs
log.thrust_mag   = zeros(K,1);
log.power_mech   = zeros(K,1);
log.energy_cum   = zeros(K,1);
log.jerk_xy      = zeros(K,2);
log.heading_rate = zeros(K,1);
log.dist_error   = zeros(K,1);

%% =========================
%  Figure + video
% =========================
fig = figure('Color','w','Name','2D MPC Chase (roll/pitch + obstacles)');
hAx = axes(fig); hold(hAx,'on'); axis(hAx,'equal'); grid(hAx,'on');
xlabel(hAx,'x (m)'); ylabel(hAx,'y (m)');

drawEnvironment(hAx, env);

hTarget = plot(hAx, xt(1), xt(2), 'r.', 'MarkerSize', 20);
hDrone  = patch(hAx, nan, nan, 'g', 'FaceAlpha', 0.35, 'EdgeColor','g', 'LineWidth', 1.5);
hLine   = plot(hAx, [xd(1) xt(1)], [xd(2) xt(2)], '-', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.0);
hText   = text(hAx, 0, 0, '', 'FontSize', 11, 'Color', 'k');

% --- MPC path overlay handles ---
% Planned drone path (cyan) and reference path (magenta dashed)
hMpcPath = plot(hAx, nan, nan, 'c-', 'LineWidth', 2.0);
hMpcDots = plot(hAx, nan, nan, 'c.', 'MarkerSize', 8);
hRefPath = plot(hAx, nan, nan, 'm--', 'LineWidth', 1.2);
hRefDots = plot(hAx, nan, nan, 'm.', 'MarkerSize', 8);

xlim(hAx, [-P.startDist-10, 40]);
ylim(hAx, [-40, 20]);

% Video writer
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

for k = 1:K
    % Log state
    log.xd(k,:) = xd';
    log.xt(k,:) = xt';
    log.phi(k) = phi;
    log.theta(k) = theta;
    log.dist(k) = norm(xt(1:2)-xd(1:2));
    log.vd(k) = norm(xd(3:4));
    log.axy(k,:) = u_prev';
    log.dist_error(k) = abs(log.dist(k) - P.R_des);

    % --- Energy computation ---
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

    % --- Jerk ---
    if k > 1
        log.jerk_xy(k,:) = (log.axy(k,:) - log.axy(k-1,:)) / P.dt;
    end

    % --- Heading rate ---
    if k > 1 && log.vd(k) > 0.5 && log.vd(k-1) > 0.5
        h_now  = atan2(xd(4), xd(3));
        h_prev = atan2(log.xd(k-1,4), log.xd(k-1,3));
        dh = atan2(sin(h_now - h_prev), cos(h_now - h_prev));
        log.heading_rate(k) = dh / P.dt;
    end

    % LoS measurement
    los = hasLineOfSight(xd(1:2), xt(1:2), env.buildings);
    log.los(k) = los;

    if los
        xthat = xt;
        xthat_last = xthat;
        lastMeasTime = t(k);
    else
        if P.usePredictionWhenNoLoS
            dtLost = t(k) - lastMeasTime;
            xthat = xthat_last;
            xthat(1:2) = xthat(1:2) + xthat(3:4)*dtLost;
        else
            xthat = xthat_last;
        end
    end

    % Reference over horizon
    xref = zeros(4, P.N+1);
    xref(:,1) = xthat;
    for i = 2:P.N+1
        xref(:,i) = xref(:,i-1);
        xref(1:2,i) = xref(1:2,i) + xref(3:4,i) * P.dt;
    end

    pref = zeros(2, P.N+1);
    vref = zeros(2, P.N+1);
    for i = 1:P.N+1
        vt_xy = xref(3:4,i);
        dir = vt_xy / (norm(vt_xy) + 1e-6);
        pref(:,i) = xref(1:2,i) - P.R_des * dir;
        vref(:,i) = xref(3:4,i);
    end

    % Solve MPC — get full sequence
    U = solveMPC_QP(xd, u_prev, pref, vref, Phi, Gamma, Du, P, W);

    if any(isnan(U))
        a_des = [0;0];
        % No valid plan: clear path overlay
        mpc_px = nan; mpc_py = nan;
    else
        a_des = U(1:2);

        % --- Reconstruct MPC predicted trajectory ---
        % X_pred = Phi * xd + Gamma * U  (stacked states over horizon)
        X_pred = Phi * xd + Gamma * U;
        mpc_px = zeros(1, P.N+1);
        mpc_py = zeros(1, P.N+1);
        for i = 1:P.N+1
            idx = (i-1)*nx + (1:nx);
            mpc_px(i) = X_pred(idx(1));
            mpc_py(i) = X_pred(idx(2));
        end
    end

    % --- Update MPC path overlay ---
    set(hMpcPath, 'XData', mpc_px, 'YData', mpc_py);
    set(hMpcDots, 'XData', mpc_px, 'YData', mpc_py);
    set(hRefPath, 'XData', pref(1,:), 'YData', pref(2,:));
    set(hRefDots, 'XData', pref(1,:), 'YData', pref(2,:));

    % Desired tilt from desired accel
    theta_cmd = atan2(a_des(1), P.g);
    phi_cmd   = -atan2(a_des(2), P.g);

    theta_cmd = clampScalar(theta_cmd, P.tiltMax);
    phi_cmd   = clampScalar(phi_cmd, P.tiltMax);

    [phi, theta] = tiltStep(phi, theta, phi_cmd, theta_cmd, P);

    % Actual accel from tilts
    accel_x = P.g * tan(theta);
    accel_y = -P.g * tan(phi);
    a_act = clamp2([accel_x; accel_y], P.amax);

    % Safety filter
    [a_safe, used] = safetyFilterAccel(xd, a_act, A, B, envInf, P);
    log.safeUsed(k) = used;

    % Propagate
    xd = A*xd + B*a_safe;
    xd(3:4) = clamp2(xd(3:4), P.vmax);

    if pointInAnyBuilding(xd(1:2), envInf.buildings)
        xd(3:4) = [0;0];
    end

    u_prev = a_safe;

    % Target step
    xt = targetStep(xt, P);

    % Draw
    set(hTarget, 'XData', xt(1), 'YData', xt(2));
    [sx, sy] = squarePatch(xd(1), xd(2), P.bladeDist);
    set(hDrone, 'XData', sx, 'YData', sy);
    set(hLine, 'XData', [xd(1) xt(1)], 'YData', [xd(2) xt(2)]);

    status = sprintf('t=%.2fs | LoS=%d | dist=%.2fm | v=%.2fm/s | E=%.1fJ | phi=%.1f%c theta=%.1f%c', ...
        t(k), los, log.dist(k), log.vd(k), energy_accum, ...
        rad2deg(phi), char(176), rad2deg(theta), char(176));

    xl = xlim(hAx);
    yl = ylim(hAx);
    xText = xl(1) + 0.02*(xl(2)-xl(1));
    yText = yl(1) + 0.90*(yl(2)-yl(1));
    set(hText, 'Position', [xText, yText, 0], 'String', status);

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
%  Compute & Print Summary
% =========================
metrics = computeMissionMetrics(t, log, P);
printMissionMetrics(metrics);

%% =========================
%  Plots: Summary Metrics
% =========================
fig_plots = figure('Color','w','Name','Performance Metrics');
% Set default font size for the figure to make text much larger
set(fig_plots, 'DefaultAxesFontSize', 16, 'DefaultTextFontSize', 16);

subplot(5,1,1);
plot(t, log.dist, 'LineWidth', 2.0); hold on;
yline(P.R_des,'--','LineWidth',1.5);
grid on; ylabel('dist (m)', 'FontWeight', 'bold'); title('Distance to target', 'FontWeight', 'bold');

subplot(5,1,2);
plot(t, rad2deg(log.phi), 'LineWidth', 2.0); hold on;
plot(t, rad2deg(log.theta), 'LineWidth', 2.0);
yline(P.tiltMaxDeg,'--','LineWidth',1.5); yline(-P.tiltMaxDeg,'--','LineWidth',1.5);
grid on; ylabel('tilt (deg)', 'FontWeight', 'bold'); title('Tilt Angles', 'FontWeight', 'bold');
legend('\phi (roll)','\theta (pitch)', 'Location', 'best');

subplot(5,1,3);
plot(t, log.power_mech, 'LineWidth', 2.0, 'Color', [0.85 0.33 0.1]);
grid on; ylabel('Power (W)', 'FontWeight', 'bold'); title('Instantaneous Power', 'FontWeight', 'bold');

subplot(5,1,4);
jerk_mag = sqrt(log.jerk_xy(:,1).^2 + log.jerk_xy(:,2).^2);
plot(t, jerk_mag, 'LineWidth', 2.0, 'Color', [0.0 0.45 0.74]);
grid on; ylabel('|jerk| (m/s^3)', 'FontWeight', 'bold'); title('Jerk Magnitude (smoothness)', 'FontWeight', 'bold');

subplot(5,1,5);
plot(t, rad2deg(log.heading_rate), 'LineWidth', 2.0, 'Color', [0.47 0.67 0.19]);
grid on; ylabel('rate (deg/s)', 'FontWeight', 'bold'); xlabel('time (s)', 'FontWeight', 'bold');
title('Velocity Heading Rate (path oscillation)', 'FontWeight', 'bold');

end

%% ========================================================================
%  Mission Metrics (local functions)
%% ========================================================================

function metrics = computeMissionMetrics(t, log, P)
K  = length(t);
dt = P.dt;

% ---- ENERGY ----
metrics.total_energy_J    = log.energy_cum(end);
metrics.total_energy_Wh   = log.energy_cum(end) / 3600;
metrics.mean_power_W      = mean(log.power_mech);
metrics.peak_power_W      = max(log.power_mech);

hover_thrust = P.mass * P.g;
hover_power  = hover_thrust^(1.5) / sqrt(2 * P.rho * P.A_disc);
hover_energy = hover_power * t(end);
metrics.hover_energy_J    = hover_energy;
metrics.hover_energy_frac = hover_energy / max(metrics.total_energy_J, 1e-6);
metrics.maneuver_energy_J = metrics.total_energy_J - hover_energy;

target_path_len = sum(sqrt(diff(log.xt(:,1)).^2 + diff(log.xt(:,2)).^2));
metrics.energy_per_meter  = metrics.total_energy_J / max(target_path_len, 1e-3);
metrics.target_path_m     = target_path_len;

% ---- OSCILLATION ----
jerk_mag = sqrt(log.jerk_xy(:,1).^2 + log.jerk_xy(:,2).^2);
metrics.jerk_rms  = rms(jerk_mag);
metrics.jerk_peak = max(jerk_mag);

metrics.heading_rate_rms_degps  = rms(rad2deg(log.heading_rate));
metrics.heading_rate_peak_degps = max(abs(rad2deg(log.heading_rate)));

dist_err = log.dist - P.R_des;
metrics.dist_error_rms  = rms(dist_err);
metrics.dist_error_peak = max(abs(dist_err));
metrics.dist_error_mean = mean(abs(dist_err));

ax_sign = sign(log.axy(:,1));
ay_sign = sign(log.axy(:,2));
metrics.accel_reversals_x     = sum(abs(diff(ax_sign)) > 0);
metrics.accel_reversals_y     = sum(abs(diff(ay_sign)) > 0);
metrics.accel_reversals_total = metrics.accel_reversals_x + metrics.accel_reversals_y;
metrics.reversal_rate_hz      = metrics.accel_reversals_total / t(end);

jerk_norm    = metrics.jerk_rms / (P.amax / dt);
heading_norm = metrics.heading_rate_rms_degps / 180;
dist_norm    = metrics.dist_error_rms / P.R_des;
rev_norm     = metrics.reversal_rate_hz / (1/dt);

metrics.oscillation_score = 0.30 * jerk_norm + ...
                            0.25 * heading_norm + ...
                            0.25 * dist_norm + ...
                            0.20 * rev_norm;
end

function printMissionMetrics(m)
fprintf('\n');
fprintf('====================================================\n');
fprintf('  MISSION METRICS SUMMARY\n');
fprintf('====================================================\n');
fprintf('\n');
fprintf('  ENERGY\n');
fprintf('  -----------------------------------------------\n');
fprintf('  Total energy:          %8.1f J  (%.4f Wh)\n', m.total_energy_J, m.total_energy_Wh);
fprintf('  Hover baseline energy: %8.1f J  (%.1f%% of total)\n', m.hover_energy_J, m.hover_energy_frac*100);
fprintf('  Maneuver energy:       %8.1f J\n', m.maneuver_energy_J);
fprintf('  Mean power:            %8.1f W\n', m.mean_power_W);
fprintf('  Peak power:            %8.1f W\n', m.peak_power_W);
fprintf('  Energy per meter:      %8.2f J/m\n', m.energy_per_meter);
fprintf('  Target path length:    %8.1f m\n', m.target_path_m);
fprintf('\n');
fprintf('  OSCILLATION\n');
fprintf('  -----------------------------------------------\n');
fprintf('  Jerk RMS:              %8.2f m/s^3\n', m.jerk_rms);
fprintf('  Jerk peak:             %8.2f m/s^3\n', m.jerk_peak);
fprintf('  Heading rate RMS:      %8.2f deg/s\n', m.heading_rate_rms_degps);
fprintf('  Heading rate peak:     %8.2f deg/s\n', m.heading_rate_peak_degps);
fprintf('  Dist error RMS:        %8.3f m\n', m.dist_error_rms);
fprintf('  Dist error peak:       %8.3f m\n', m.dist_error_peak);
fprintf('  Dist error mean:       %8.3f m\n', m.dist_error_mean);
fprintf('  Accel reversals (x/y): %d / %d  (total %d)\n', ...
    m.accel_reversals_x, m.accel_reversals_y, m.accel_reversals_total);
fprintf('  Reversal rate:         %8.2f Hz\n', m.reversal_rate_hz);
fprintf('\n');
fprintf('  COMPOSITE OSCILLATION SCORE: %.4f\n', m.oscillation_score);
fprintf('    (0 = perfect smoothness, lower is better)\n');
fprintf('====================================================\n\n');
end