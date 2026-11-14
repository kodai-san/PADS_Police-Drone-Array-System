function simulate_mpc_chase_2D_rollpitch_video()
% 2D MPC chase simulation with:
% - roll/pitch dynamics (first-order tilt response)
% - tilt limit (±30 deg)
% - hard building obstacles (drone cannot enter)
% - LoS measurement dropouts due to buildings
% - MP4 video output
%
% Target is a generic moving point on road centerline (test dummy / cooperative tag).
%
% Requires: Optimization Toolbox (quadprog)

%% =========================
%  Parameters
% =========================
P.dt        = 1/20;     % 20 Hz
P.Tsim      = 25.0;     % seconds
P.N         = 20;       % horizon (1.0s)

% Geometry (visual only)
P.bladeDist = 0.40;     % meters (green square side)

% Limits
P.vmax      = 20.0;     % m/s (horizontal)
P.amax      = 5.6;      % m/s^2 (horizontal)
P.tiltMaxDeg= 30.0;     % deg
P.tiltMax   = deg2rad(P.tiltMaxDeg);
P.g         = 9.81;

% Tilt dynamics (tunable)
P.tiltTau   = 0.25;     % seconds (1st order response)
P.tiltRateMax = deg2rad(180); % rad/s clamp

% Target
P.vt        = 3.0;      % m/s
P.roadW     = 10.0;     % m
P.startDist = 20.0;     % m before intersection
P.turnDirection = "south"; % "south" or "north"

% Tracking objective: stay behind target by R_des along target velocity
P.R_des     = 5.0;

% LoS / estimator
P.usePredictionWhenNoLoS = true;

% MPC weights (tunable)
W.w_pos = 10.0;
W.w_vel = 2.0;
W.w_u   = 2;
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
xd = [xt(1)-3.0; xt(2); 0; 0];

% Drone attitude states (roll phi, pitch theta)
phi = 0; theta = 0;

% Estimator memory (for target)
xthat = xt;
xthat_last = xt;
lastMeasTime = 0;

% Environment
env = makeIntersectionEnvironment(P.roadW);
envInf = inflateBuildings(env, P.obstacleInflation);

% MPC discrete planar model (we optimize acceleration commands u=[ax; ay])
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
u_prev = [0;0]; % accel cmd used by MPC (desired accel)
for k = 1:K
    % Log
    log.xd(k,:) = xd';
    log.xt(k,:) = xt';
    log.phi(k) = phi;
    log.theta(k) = theta;
    log.dist(k) = norm(xt(1:2)-xd(1:2));
    log.vd(k) = norm(xd(3:4));
    log.axy(k,:) = u_prev';

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

    % Reference over horizon (use road model)
    xref = zeros(4, P.N+1);
    xref(:,1) = xthat;
    for i = 2:P.N+1
        xref(:,i) = targetStep(xref(:,i-1), P);
    end

    pref = zeros(2, P.N+1);
    vref = zeros(2, P.N+1);
    for i = 1:P.N+1
        vt_xy = xref(3:4,i);
        dir = vt_xy / (norm(vt_xy) + 1e-6);
        pref(:,i) = xref(1:2,i) - P.R_des * dir;  % behind target
        vref(:,i) = xref(3:4,i);
    end

    % Solve planar MPC for desired accel sequence
    U = solveMPC_QP(xd, u_prev, pref, vref, Phi, Gamma, Du, P, W);

    if any(isnan(U))
        a_des = [0;0];
    else
        a_des = U(1:2);
    end

    % Convert desired accel -> desired tilt
    % pitch theta controls +x acceleration, roll phi controls +y acceleration
    theta_cmd = atan2(a_des(1), P.g);
    phi_cmd   = -atan2(a_des(2), P.g);

    % Tilt limits
    theta_cmd = clampScalar(theta_cmd, P.tiltMax);
    phi_cmd   = clampScalar(phi_cmd, P.tiltMax);

    % Tilt dynamics (1st order + rate clamp)
    [phi, theta] = tiltStep(phi, theta, phi_cmd, theta_cmd, P);

    % Actual horizontal accel from current tilts
    accel_x = P.g * tan(theta);
    accel_y = -P.g * tan(phi);
    a_act = [accel_x; accel_y];

    % Enforce accel magnitude cap
    a_act = clamp2(a_act, P.amax);

    % Safety filter: ensure next step doesn't enter inflated buildings
    [a_safe, used] = safetyFilterAccel(xd, a_act, A, B, envInf, P);
    log.safeUsed(k) = used;

    % Propagate drone one step with safe accel
    xd = A*xd + B*a_safe;

    % Speed limit
    xd(3:4) = clamp2(xd(3:4), P.vmax);

    % If somehow inside building, emergency stop
    if pointInAnyBuilding(xd(1:2), envInf.buildings)
        xd(3:4) = [0;0];
    end

    % Update "previous command" for MPC smoothing
    u_prev = a_safe;

    % True target step
    xt = targetStep(xt, P);

    % Draw
    set(hTarget, 'XData', xt(1), 'YData', xt(2));
    [sx, sy] = squarePatch(xd(1), xd(2), P.bladeDist);
    set(hDrone, 'XData', sx, 'YData', sy);
    set(hLine, 'XData', [xd(1) xt(1)], 'YData', [xd(2) xt(2)]);

    status = sprintf('t=%.2fs | LoS=%d | dist=%.2fm | v=%.2fm/s | phi=%.1f° theta=%.1f° | safety=%d', ...
        t(k), los, log.dist(k), log.vd(k), rad2deg(phi), rad2deg(theta), used);

    xl = xlim(hAx);
    yl = ylim(hAx);
    xText = xl(1) + 0.02*(xl(2)-xl(1));
    yText = yl(1) + 0.90*(yl(2)-yl(1));
    set(hText, 'Position', [xText, yText, 0], 'String', status);

    drawnow limitrate;

    % Write video frame
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
%  Post-sim Plots
% =========================
figure('Color','w','Name','Metrics');

subplot(5,1,1);
plot(t, log.dist, 'LineWidth', 1.5); hold on;
yline(P.R_des,'--','LineWidth',1.0);
grid on; ylabel('dist (m)'); title('Distance to target');

subplot(5,1,2);
plot(t, log.vd, 'LineWidth', 1.5); grid on;
ylabel('speed (m/s)');

subplot(5,1,3);
plot(t, log.axy(:,1), 'LineWidth', 1.2); hold on;
plot(t, log.axy(:,2), 'LineWidth', 1.2);
yline(P.amax,'--'); yline(-P.amax,'--');
grid on; ylabel('a (m/s^2)'); legend('ax','ay');

subplot(5,1,4);
plot(t, rad2deg(log.phi), 'LineWidth', 1.2); hold on;
plot(t, rad2deg(log.theta), 'LineWidth', 1.2);
yline(P.tiltMaxDeg,'--'); yline(-P.tiltMaxDeg,'--');
grid on; ylabel('tilt (deg)'); legend('\phi (roll)','\theta (pitch)');

subplot(5,1,5);
stairs(t, log.los, 'LineWidth', 1.2); hold on;
stairs(t, log.safeUsed, 'LineWidth', 1.2);
grid on; ylim([-0.1 1.1]);
ylabel('flags'); xlabel('time (s)');
legend('LoS','Safety used');

end
