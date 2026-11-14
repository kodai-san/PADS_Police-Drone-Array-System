function front_camera_view(logfile)
% front_camera_view  Render a synthetic front-camera view from logged sim data.
%
% This takes the 2D simulation and projects the scene (buildings, road edges,
% target) into a perspective "forward camera" display as if a camera were
% mounted on the drone pointing along its velocity vector.
%
% Usage:
%   1. Run simulate_mpc_chase_2D_rollpitch_video first (saves workspace)
%   2. Call:  front_camera_view('mpc_chase_log.mat')
%
%   Or call with no arguments to run an embedded simulation and generate
%   the camera view directly.
%
% The view is synthetic / illustrative — the sim is 2D so we invent a
% flight altitude and use simple pinhole projection to show:
%   - Road edge lines receding into the distance
%   - Building walls as filled rectangles
%   - Target as a marker
%   - Horizon line
%   - Roll / pitch visualised as camera tilt
%
% Output: MP4 video 'front_camera_view.mp4'

if nargin < 1
    % Run embedded simulation to get log data
    [t, log, P, env] = runSimForCamera();
else
    S = load(logfile);
    t   = S.t;
    log = S.log;
    P   = S.P;
    env = S.env;
end

%% Camera parameters
cam.alt     = 8.0;    % flight altitude (m) — invented for 3D projection
cam.fov_deg = 90;     % horizontal field of view (degrees)
cam.fov     = deg2rad(cam.fov_deg);
cam.imgW    = 640;     % image width (pixels)
cam.imgH    = 480;     % image height (pixels)
cam.f       = (cam.imgW/2) / tan(cam.fov/2);  % focal length (pixels)
cam.cx      = cam.imgW / 2;
cam.cy      = cam.imgH / 2;
cam.maxRange = 50;    % max rendering distance (m)

%% Setup figure
fig = figure('Color','k','Name','Front Camera View', ...
    'Position', [100 100 cam.imgW cam.imgH]);
hAx = axes(fig, 'Color', [0.55 0.75 0.95]); % sky blue background
hold(hAx, 'on');
set(hAx, 'XLim', [0 cam.imgW], 'YLim', [0 cam.imgH]);
set(hAx, 'YDir', 'reverse');  % image coordinates: y down
set(hAx, 'XTick', [], 'YTick', []);
axis(hAx, 'off');

% Static elements (redrawn each frame)
hHorizon = plot(hAx, [0 cam.imgW], [cam.cy cam.cy], '--', ...
    'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);

% Preallocate graphics handles
hGround   = patch(hAx, [0 cam.imgW cam.imgW 0], ...
    [cam.cy cam.cy cam.imgH cam.imgH], ...
    [0.4 0.4 0.35], 'EdgeColor', 'none');
hRoadL    = plot(hAx, nan, nan, 'w-', 'LineWidth', 2);
hRoadR    = plot(hAx, nan, nan, 'w-', 'LineWidth', 2);
hRoadCtr  = plot(hAx, nan, nan, 'w--', 'LineWidth', 1);
hBuildings = gobjects(0);
hTargetFwd = plot(hAx, nan, nan, 'r.', 'MarkerSize', 30);
hTargetBox = plot(hAx, nan, nan, 'rs', 'MarkerSize', 20, 'LineWidth', 2);
hCrosshair = plot(hAx, cam.cx, cam.cy, 'g+', 'MarkerSize', 15, 'LineWidth', 1.5);
hHUD       = text(hAx, 10, 20, '', 'Color', 'g', 'FontSize', 10, ...
    'FontName', 'FixedWidth', 'FontWeight', 'bold');
hHUD2      = text(hAx, 10, cam.imgH-30, '', 'Color', [0.8 0.8 0.8], ...
    'FontSize', 9, 'FontName', 'FixedWidth');

% Video writer
vw = VideoWriter('front_camera_view.mp4', 'MPEG-4');
vw.FrameRate = 20;
open(vw);

%% Render loop
K = length(t);
roadHW = env.roadHalfW;

for k = 1:K
    % Drone state
    dx = log.xd(k,1); dy = log.xd(k,2);
    dvx = log.xd(k,3); dvy = log.xd(k,4);
    phi_k   = log.phi(k);
    theta_k = log.theta(k);

    % Target state
    tx = log.xt(k,1); ty = log.xt(k,2);

    % Drone heading (velocity direction, or default +x if stopped)
    speed = sqrt(dvx^2 + dvy^2);
    if speed > 0.3
        heading = atan2(dvy, dvx);
    else
        heading = 0;
    end

    % Camera rotation matrix (world -> camera frame)
    % Camera frame: +Z = forward (along heading), +X = right, +Y = down
    ch = cos(heading); sh = sin(heading);

    % Clear dynamic building patches from previous frame
    delete(hBuildings);
    hBuildings = gobjects(0);

    % ---- Project road edges ----
    % Sample points along the road ahead of the drone
    nPts = 40;
    dists = linspace(1, cam.maxRange, nPts);

    roadL_u = nan(1, nPts);  roadL_v = nan(1, nPts);
    roadR_u = nan(1, nPts);  roadR_v = nan(1, nPts);
    roadC_u = nan(1, nPts);  roadC_v = nan(1, nPts);

    for i = 1:nPts
        % World point along forward direction at distance dists(i)
        % Road edges: offset perpendicular to road direction
        % For simplicity, determine which road segment we're on
        fwd_world = [dx + dists(i)*ch; dy + dists(i)*sh];

        % Determine road edges at this forward point
        % Horizontal road: y = ±roadHW  (if |y| < roadHW means on horiz road)
        % Vertical road:   x = ±roadHW  (if |x| < roadHW means on vert road)
        [leftEdge, rightEdge, ctrLine] = getRoadEdgesAt(fwd_world, roadHW, heading);

        % Project left edge into camera image
        [u, v, vis] = projectToCamera(leftEdge, [dx; dy], heading, cam.alt, phi_k, theta_k, cam);
        if vis, roadL_u(i) = u; roadL_v(i) = v; end

        [u, v, vis] = projectToCamera(rightEdge, [dx; dy], heading, cam.alt, phi_k, theta_k, cam);
        if vis, roadR_u(i) = u; roadR_v(i) = v; end

        [u, v, vis] = projectToCamera(ctrLine, [dx; dy], heading, cam.alt, phi_k, theta_k, cam);
        if vis, roadC_u(i) = u; roadC_v(i) = v; end
    end

    set(hRoadL, 'XData', roadL_u, 'YData', roadL_v);
    set(hRoadR, 'XData', roadR_u, 'YData', roadR_v);
    set(hRoadCtr, 'XData', roadC_u, 'YData', roadC_v);

    % ---- Project buildings ----
    for bi = 1:numel(env.buildings)
        bldg = env.buildings{bi};
        nb = size(bldg, 1);
        bu = nan(1, nb); bv = nan(1, nb);
        anyVis = false;
        for j = 1:nb
            wp = bldg(j, :)';
            [u, v, vis] = projectToCamera(wp, [dx;dy], heading, cam.alt, phi_k, theta_k, cam);
            if vis
                bu(j) = u; bv(j) = v;
                anyVis = true;
            end
        end
        if anyVis
            % Fill visible building faces
            validIdx = ~isnan(bu);
            if sum(validIdx) >= 3
                hp = patch(hAx, bu(validIdx), bv(validIdx), [0.6 0.6 0.65], ...
                    'EdgeColor', [0.4 0.4 0.45], 'LineWidth', 1, 'FaceAlpha', 0.8);
                hBuildings(end+1) = hp; %#ok<AGROW>
            end
        end
    end

    % ---- Project target ----
    [tu, tv, tvis] = projectToCamera([tx; ty], [dx; dy], heading, cam.alt, phi_k, theta_k, cam);
    if tvis
        set(hTargetFwd, 'XData', tu, 'YData', tv, 'Visible', 'on');
        set(hTargetBox, 'XData', tu, 'YData', tv, 'Visible', 'on');
    else
        set(hTargetFwd, 'Visible', 'off');
        set(hTargetBox, 'Visible', 'off');
    end

    % ---- Update horizon line with roll ----
    % Roll tilts the horizon
    roll_px = cam.imgW * 0.6 * sin(phi_k);
    set(hHorizon, 'YData', [cam.cy + roll_px, cam.cy - roll_px]);

    % ---- Update ground color (below horizon) ----
    gndY = cam.cy + roll_px;
    gndY2 = cam.cy - roll_px;
    set(hGround, 'YData', [min(gndY,gndY2), max(gndY,gndY2), cam.imgH, cam.imgH]);

    % ---- HUD overlay ----
    dist_to_target = norm([tx-dx, ty-dy]);
    bearing_to_target = rad2deg(atan2(ty-dy, tx-dx) - heading);
    bearing_to_target = mod(bearing_to_target + 180, 360) - 180;

    hudStr = sprintf('SPD %4.1f m/s  ALT %4.1f m\nTGT %5.1f m  BRG %+4.0f%c\nLoS %d  SAF %d', ...
        speed, cam.alt, dist_to_target, bearing_to_target, char(176), ...
        log.los(k), log.safeUsed(k));
    set(hHUD, 'String', hudStr);

    attStr = sprintf('ROLL %+5.1f%c  PITCH %+5.1f%c  t=%.2fs', ...
        rad2deg(phi_k), char(176), rad2deg(theta_k), char(176), t(k));
    set(hHUD2, 'String', attStr);

    drawnow limitrate;
    frame = getframe(fig);
    writeVideo(vw, frame);
end

close(vw);
fprintf('Saved front camera video to: front_camera_view.mp4\n');
end

%% ========================================================================
%  Projection helper
%% ========================================================================
function [u, v, visible] = projectToCamera(worldPt2D, camPos2D, heading, alt, phi, theta, cam)
% Project a 2D world point into the camera image.
%
% Camera is at (camPos2D, alt) looking along heading direction.
% Simple pinhole model with roll (phi) and pitch (theta) applied.

% Vector from camera to point in world frame
dw = worldPt2D(:) - camPos2D(:);

% Rotate into camera frame (heading-aligned)
ch = cos(-heading); sh = sin(-heading);
R2 = [ch -sh; sh ch];
dc = R2 * dw;   % dc(1) = forward, dc(2) = right

fwd   = dc(1);
right = dc(2);
down  = alt;     % point is on ground, camera is at altitude

% Apply pitch rotation (positive theta = nose down = see more ground)
cp = cos(theta); sp = sin(theta);
fwd2  = cp*fwd + sp*down;
down2 = -sp*fwd + cp*down;

% Apply roll rotation (positive phi = right wing down)
cr = cos(phi); sr = sin(phi);
right2 = cr*right - sr*down2;
down3  = sr*right + cr*down2;

% Pinhole projection (Z = forward)
visible = (fwd2 > 0.5);  % must be in front of camera
if ~visible
    u = nan; v = nan;
    return;
end

u = cam.cx + cam.f * (right2 / fwd2);
v = cam.cy + cam.f * (down3  / fwd2);

% Clip to image bounds (with margin)
margin = 200;
if u < -margin || u > cam.imgW + margin || v < -margin || v > cam.imgH + margin
    visible = false;
    u = nan; v = nan;
end
end

%% ========================================================================
%  Road edge helper
%% ========================================================================
function [leftPt, rightPt, ctrPt] = getRoadEdgesAt(worldPt, roadHW, heading)
% Given a world point and drone heading, return the nearest road edge
% points (left and right of the road the drone is likely following).

px = worldPt(1); py = worldPt(2);

% Determine which road segment this point is nearest to
onHorizRoad = (abs(py) <= roadHW * 1.5);
onVertRoad  = (abs(px) <= roadHW * 1.5);

% Heading direction: mostly horizontal or vertical?
absH = abs(cos(heading));

if onHorizRoad && (absH > 0.5 || ~onVertRoad)
    % Treat as horizontal road
    leftPt  = [px;  roadHW];
    rightPt = [px; -roadHW];
    ctrPt   = [px;  0];
elseif onVertRoad
    % Treat as vertical road
    leftPt  = [-roadHW; py];
    rightPt = [ roadHW; py];
    ctrPt   = [0;       py];
else
    % Off-road: just put edges far away (won't render well)
    leftPt  = [px; py + roadHW];
    rightPt = [px; py - roadHW];
    ctrPt   = [px; py];
end
end

%% ========================================================================
%  Embedded mini-simulation (reuses the project's functions)
%% ========================================================================
function [t, log, P, env] = runSimForCamera()
% Minimal re-run of the simulation to capture log data.
% (Identical physics to the main sim, but no figure/video output.)

P.dt        = 1/20;
P.Tsim      = 25.0;
P.N         = 20;
P.bladeDist = 0.40;
P.vmax      = 20.0;
P.amax      = 5.6;
P.tiltMaxDeg= 30.0;
P.tiltMax   = deg2rad(P.tiltMaxDeg);
P.g         = 9.81;
P.tiltTau   = 0.25;
P.tiltRateMax = deg2rad(180);
P.mass      = 2.5;
P.rho       = 1.225;
P.A_disc    = 0.04;
P.vt        = 3.0;
P.roadW     = 10.0;
P.startDist = 30.0;
P.turnDirection = "south";
P.R_des     = 5.0;
P.usePredictionWhenNoLoS = true;
P.obstacleInflation = 0.5;
P.safetyCandidates  = 16;
P.safeStepLookahead = 1;
P.emergencyBrake    = true;

W.w_pos = 60.0;  W.w_vel = 2.0;
W.w_u   = 0.8;   W.w_du  = 3.0;

K = floor(P.Tsim/P.dt) + 1;
t = (0:K-1)'*P.dt;

xt = [-P.startDist; 0; P.vt; 0];
xd = [xt(1)-3.0; xt(2); 0; 0];
phi = 0; theta = 0;
xthat = xt; xthat_last = xt; lastMeasTime = 0;

env = makeIntersectionEnvironment(P.roadW);
envInf = inflateBuildings(env, P.obstacleInflation);

Amat = [1 0 P.dt 0; 0 1 0 P.dt; 0 0 1 0; 0 0 0 1];
Bmat = [0.5*P.dt^2 0; 0 0.5*P.dt^2; P.dt 0; 0 P.dt];
[Phi, Gamma] = buildPredictionMatrices(Amat, Bmat, P.N);
Du = diffMatrix2D(P.N);

log.xd  = zeros(K,4);
log.xt  = zeros(K,4);
log.phi = zeros(K,1);
log.theta = zeros(K,1);
log.los = false(K,1);
log.safeUsed = false(K,1);

u_prev = [0;0];

for k = 1:K
    log.xd(k,:) = xd';
    log.xt(k,:) = xt';
    log.phi(k)  = phi;
    log.theta(k) = theta;

    los = hasLineOfSight(xd(1:2), xt(1:2), env.buildings);
    log.los(k) = los;

    if los
        xthat = xt; xthat_last = xthat; lastMeasTime = t(k);
    else
        if P.usePredictionWhenNoLoS
            dtLost = t(k) - lastMeasTime;
            xthat = xthat_last;
            xthat(1:2) = xthat(1:2) + xthat(3:4)*dtLost;
        else
            xthat = xthat_last;
        end
    end

    xref = zeros(4, P.N+1); xref(:,1) = xthat;
    for i = 2:P.N+1, xref(:,i) = targetStep(xref(:,i-1), P); end

    pref = zeros(2, P.N+1); vref = zeros(2, P.N+1);
    for i = 1:P.N+1
        dir = xref(3:4,i)/(norm(xref(3:4,i))+1e-6);
        pref(:,i) = xref(1:2,i) - P.R_des*dir;
        vref(:,i) = xref(3:4,i);
    end

    U = solveMPC_QP(xd, u_prev, pref, vref, Phi, Gamma, Du, P, W);
    if any(isnan(U)), a_des=[0;0]; else, a_des=U(1:2); end

    theta_cmd = clampScalar(atan2(a_des(1),P.g), P.tiltMax);
    phi_cmd   = clampScalar(-atan2(a_des(2),P.g), P.tiltMax);
    [phi, theta] = tiltStep(phi, theta, phi_cmd, theta_cmd, P);

    a_act = clamp2([P.g*tan(theta); -P.g*tan(phi)], P.amax);
    [a_safe, used] = safetyFilterAccel(xd, a_act, Amat, Bmat, envInf, P);
    log.safeUsed(k) = used;

    xd = Amat*xd + Bmat*a_safe;
    xd(3:4) = clamp2(xd(3:4), P.vmax);
    if pointInAnyBuilding(xd(1:2), envInf.buildings), xd(3:4)=[0;0]; end

    u_prev = a_safe;
    xt = targetStep(xt, P);
end

fprintf('Headless simulation complete (%d steps).\n', K);
end
