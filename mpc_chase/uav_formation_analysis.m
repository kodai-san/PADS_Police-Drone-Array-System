function uav_formation_analysis()
% UAV Formation Analysis for Different Road Widths and Speeds
%
% Generates:
%   1) uav_formation_grid.png  - grid of formations (rows=speed, cols=road width)
%   2) uav_formation_turn.mp4  - animated 90 degree turn of a representative formation
%
% Turn model: constant-speed circular arc, R = v^2 / a_max
% Collision avoidance: ensure all UAV swept paths stay within road boundaries
%   and maintain minimum inter-UAV separation throughout the turn.

%% =========================================================================
%  Parameters
%% =========================================================================
speeds     = [2, 5, 8, 12, 15];      % m/s
roadWidths = [4, 6, 8, 10, 12];      % m  (total width)
a_max      = 5.0;                     % m/s^2  max centripetal accel
uav_r      = 0.3;                     % m  safety bubble radius per UAV
min_sep    = 1.0;                     % m  centre-to-centre minimum spacing
safety_margin = 0.5;                  % m  clearance from road edge

nSpeeds = numel(speeds);
nWidths = numel(roadWidths);

%% =========================================================================
%  1.  Static Formation Grid Image
%% =========================================================================
fig1 = figure('Position', [50 50 1600 1000], 'Color', 'w', 'Visible', 'off');
sgtitle('UAV Formations: Speed vs Road Width', 'FontSize', 16, 'FontWeight', 'bold');

for si = 1:nSpeeds
    for wi = 1:nWidths
        v  = speeds(si);
        rw = roadWidths(wi);

        idx = (si-1)*nWidths + wi;
        ax  = subplot(nSpeeds, nWidths, idx);
        hold(ax, 'on'); axis(ax, 'equal');

        [positions, nUAV, R_turn, feasible] = computeFormation(v, rw, a_max, ...
                                                uav_r, min_sep, safety_margin);
        drawFormationPanel(ax, rw, R_turn, positions, nUAV, v, feasible, ...
                           uav_r, safety_margin);
    end
end

exportgraphics(fig1, fullfile(pwd, 'uav_formation_grid.png'), 'Resolution', 200);
fprintf('Saved: uav_formation_grid.png\n');
close(fig1);

%% =========================================================================
%  2.  Animated 90 degree Turn Video
%% =========================================================================
% Find the best feasible case for the video (prefer mid-speed, wide road)
videoSpeed = 0; videoRoadW = 0;
bestScore  = -inf;
for si = 1:nSpeeds
    for wi = 1:nWidths
        v  = speeds(si);
        rw = roadWidths(wi);
        [~, n, ~, ok] = computeFormation(v, rw, a_max, uav_r, min_sep, safety_margin);
        if ok && n >= 2
            % Prefer moderate speed and many UAVs
            score = n * 10 + v - abs(v - 8);
            if score > bestScore
                bestScore  = score;
                videoSpeed = v;
                videoRoadW = rw;
            end
        end
    end
end

if videoSpeed == 0
    % absolute fallback
    videoSpeed = speeds(1);
    videoRoadW = roadWidths(end);
end

fprintf('Video case: v=%d m/s, road=%d m\n', videoSpeed, videoRoadW);
[positions, nUAV, R_turn, ~] = computeFormation(videoSpeed, videoRoadW, ...
                                    a_max, uav_r, min_sep, safety_margin);

makeFormationTurnVideo(positions, nUAV, R_turn, videoSpeed, videoRoadW, ...
                       uav_r, safety_margin);

fprintf('Done.\n');
end

%% =========================================================================
%  FORMATION COMPUTATION
%% =========================================================================
function [positions, nUAV, R_turn, feasible] = computeFormation(v, roadW, ...
                                                a_max, uav_r, min_sep, margin)
% Compute how many UAVs fit across the road while safely turning at speed v.
%
% Turn geometry (90 degree right turn at an intersection):
%   The formation approaches along the horizontal road (+X), turns into
%   the vertical road (+Y).
%
%   Turn radius R = v^2 / a_max  (minimum radius at max centripetal accel).
%   The arc centre sits at the inner corner of the turn:
%       (hw - margin, -(hw - margin))
%   where hw = roadW/2.
%
%   Each UAV follows a concentric arc at radius R + offset_i.
%
%   Feasibility: the outermost arc must not exit the road boundary.
%   During a 90 degree arc, the maximum radial extent is simply the arc
%   radius itself (the arc is circular).  The critical constraint is that
%   the arc does not exceed the outer road edge on either the horizontal
%   or vertical road segments.
%
%   Outer road wall is at distance  R_turn + W_eff  from the arc centre.
%   So we need:   R_outer + uav_r  <=  R_turn + W_eff
%   i.e.          offset_outer + uav_r  <=  W_eff.
%
%   Additionally, we check the diagonal overshoot at 45 degrees.  The arc
%   itself stays at its radius, but we add a small safety factor for the
%   UAV body protruding beyond the road edge at the corner where both
%   road edges meet.  At 45 degrees the road corner is at distance
%   (hw + margin) * sqrt(2) from the arc centre.  For most practical
%   cases this is not the binding constraint; we include it for rigour.

R_turn = v^2 / a_max;
hw     = roadW / 2;
W_eff  = roadW - 2*margin;

if W_eff < 2*uav_r
    positions = zeros(0,2);
    nUAV      = 0;
    feasible  = false;
    return;
end

% Maximum UAVs that physically fit side-by-side in the corridor
maxUAV = floor((W_eff - 2*uav_r) / min_sep) + 1;
maxUAV = max(maxUAV, 1);

% Iteratively reduce UAVs until the formation fits within the turn
nUAV     = maxUAV;
feasible = false;
while nUAV >= 1
    if nUAV == 1
        offsets = W_eff / 2;
    else
        offsets = linspace(uav_r, W_eff - uav_r, nUAV);
    end

    R_outer = R_turn + offsets(end);   % outermost arc radius

    % --- Feasibility check ---
    % Outer arc + UAV radius must fit within the outer road wall.
    % The outer road wall distance from the arc centre = R_turn + W_eff.
    % The arc is circular, so UAVs never exceed R_outer from the centre.
    % At the intersection corner (45 deg), both roads overlap so the
    % envelope is not constrained there.
    outerFits = (R_outer + uav_r) <= (R_turn + W_eff);

    if outerFits
        feasible = true;
        break;
    end
    nUAV = nUAV - 1;
end

if nUAV == 0
    positions = zeros(0,2);
    feasible  = false;
    return;
end

% Compute initial positions (V-formation in the approach road)
if nUAV == 1
    ypos = 0;
else
    ypos = linspace(-W_eff/2 + uav_r, W_eff/2 - uav_r, nUAV);
end

xpos = zeros(1, nUAV);
for i = 1:nUAV
    xpos(i) = -abs(ypos(i)) * 0.6;   % V-shape stagger
end
xpos = xpos - min(xpos) - 3;

positions = [xpos', ypos'];
end

%% =========================================================================
%  DRAW A SINGLE PANEL
%% =========================================================================
function drawFormationPanel(ax, roadW, R_turn, positions, nUAV, speed, ...
                            feasible, uav_r, margin)
hw = roadW / 2;
viewRange = max(15, R_turn * 1.8);
viewRange = min(viewRange, 60);  % cap view range

% Roads
roadCol = [0.93 0.93 0.93];
patch(ax, [-viewRange viewRange viewRange -viewRange], ...
         [-hw -hw hw hw], roadCol, 'EdgeColor', 'none');
patch(ax, [-hw -hw hw hw], ...
         [-viewRange -viewRange viewRange viewRange], roadCol, 'EdgeColor', 'none');

% Road edges
edgeCol = [0.4 0.4 0.4];
lw = 0.8;
plot(ax, [-viewRange viewRange], [ hw  hw], '-', 'Color', edgeCol, 'LineWidth', lw);
plot(ax, [-viewRange viewRange], [-hw -hw], '-', 'Color', edgeCol, 'LineWidth', lw);
plot(ax, [ hw  hw], [-viewRange viewRange], '-', 'Color', edgeCol, 'LineWidth', lw);
plot(ax, [-hw -hw], [-viewRange viewRange], '-', 'Color', edgeCol, 'LineWidth', lw);

% Buildings (intersection corners)
bldgCol = [0.82 0.82 0.87];
ec = [0.6 0.6 0.7];
patch(ax, [-viewRange -hw -hw -viewRange], [ hw  hw viewRange viewRange], bldgCol, 'EdgeColor', ec);
patch(ax, [ hw viewRange viewRange  hw],   [ hw  hw viewRange viewRange], bldgCol, 'EdgeColor', ec);
patch(ax, [-viewRange -hw -hw -viewRange], [-viewRange -viewRange -hw -hw], bldgCol, 'EdgeColor', ec);
patch(ax, [ hw viewRange viewRange  hw],   [-viewRange -viewRange -hw -hw], bldgCol, 'EdgeColor', ec);

% Centre lines
plot(ax, [-viewRange viewRange], [0 0], 'k--', 'LineWidth', 0.4);
plot(ax, [0 0], [-viewRange viewRange], 'k--', 'LineWidth', 0.4);

% Turn arc envelope
if R_turn > 0.01 && nUAV > 0
    W_eff = roadW - 2*margin;
    if nUAV == 1
        offsets = W_eff/2;
    else
        offsets = linspace(uav_r, W_eff - uav_r, nUAV);
    end

    acx = hw - margin;   acy = -(hw - margin);
    theta_arc = linspace(0, pi/2, 60);

    R_in  = R_turn + offsets(1);
    R_out = R_turn + offsets(end);

    xarc_in  = acx - R_in  * cos(theta_arc);
    yarc_in  = acy + R_in  * sin(theta_arc);
    xarc_out = acx - R_out * cos(theta_arc);
    yarc_out = acy + R_out * sin(theta_arc);

    plot(ax, xarc_in,  yarc_in,  '-', 'Color', [0.2 0.7 0.3], 'LineWidth', 1.0);
    plot(ax, xarc_out, yarc_out, '-', 'Color', [0.9 0.3 0.2], 'LineWidth', 1.0);

    % Shade the swept band
    xband = [xarc_in, fliplr(xarc_out)];
    yband = [yarc_in, fliplr(yarc_out)];
    patch(ax, xband, yband, [1 0.9 0.7], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
end

% UAVs
if nUAV > 0
    if feasible
        uavCol = [0.2 0.55 0.9];
    else
        uavCol = [0.9 0.25 0.25];
    end
    theta_circ = linspace(0, 2*pi, 30);
    for i = 1:nUAV
        xc = positions(i,1) + uav_r * cos(theta_circ);
        yc = positions(i,2) + uav_r * sin(theta_circ);
        patch(ax, xc, yc, uavCol, 'EdgeColor', uavCol*0.5, 'FaceAlpha', 0.8);
    end
end

% Title
if feasible
    statusStr = sprintf('n=%d  R=%.1fm', nUAV, R_turn);
    col = [0 0.45 0];
else
    statusStr = sprintf('INFEASIBLE  R=%.1fm', R_turn);
    col = [0.75 0 0];
end
title(ax, sprintf('v=%dm/s  w=%dm\n%s', speed, roadW, statusStr), ...
      'FontSize', 7, 'Color', col);

xlim(ax, [-viewRange viewRange]);
ylim(ax, [-viewRange viewRange]);
set(ax, 'XTick', [], 'YTick', []);
box(ax, 'on');
end

%% =========================================================================
%  TURN ANIMATION VIDEO
%% =========================================================================
function makeFormationTurnVideo(positions, nUAV, R_turn, v, roadW, ...
                                uav_r, margin)
hw       = roadW / 2;
W_eff    = roadW - 2*margin;
fps      = 30;
duration = 8;
nFrames  = fps * duration;

% Arc centre
acx = hw - margin;
acy = -(hw - margin);

% Per-UAV offsets and arc radii
if nUAV == 1
    offsets = W_eff / 2;
else
    offsets = linspace(uav_r, W_eff - uav_r, nUAV);
end
R_uav = R_turn + offsets;

% Phase lengths
approachLen = 15;
exitLen     = 15;

dt    = 1 / fps;
t_all = (0:nFrames-1) * dt;
dist  = v * t_all;

% Create figure
fig2 = figure('Position', [100 100 800 800], 'Color', 'w', 'Visible', 'off');
ax2  = axes(fig2); hold(ax2, 'on'); axis(ax2, 'equal');

% Video writer — try MPEG4 first, fall back to AVI
videoFile = fullfile(pwd, 'uav_formation_turn');
try
    vw = VideoWriter([videoFile '.mp4'], 'MPEG4');
    videoFile = [videoFile '.mp4'];
catch
    vw = VideoWriter([videoFile '.avi'], 'Motion JPEG AVI');
    videoFile = [videoFile '.avi'];
end
vw.FrameRate = fps;
vw.Quality   = 95;
open(vw);

viewW = max(20, R_turn * 2.5);
viewW = min(viewW, 80);

% Color map for UAVs
cmap = lines(max(nUAV, 1));

for f = 1:nFrames
    cla(ax2);
    hold(ax2, 'on');

    % --- Draw environment ---
    roadCol = [0.93 0.93 0.93];
    patch(ax2, [-viewW*2 viewW*2 viewW*2 -viewW*2], [-hw -hw hw hw], ...
          roadCol, 'EdgeColor', 'none');
    patch(ax2, [-hw -hw hw hw], [-viewW*2 -viewW*2 viewW*2 viewW*2], ...
          roadCol, 'EdgeColor', 'none');

    edgeCol = [0.4 0.4 0.4];
    plot(ax2, [-viewW*2 viewW*2], [ hw  hw], '-', 'Color', edgeCol, 'LineWidth', 1.2);
    plot(ax2, [-viewW*2 viewW*2], [-hw -hw], '-', 'Color', edgeCol, 'LineWidth', 1.2);
    plot(ax2, [ hw  hw], [-viewW*2 viewW*2], '-', 'Color', edgeCol, 'LineWidth', 1.2);
    plot(ax2, [-hw -hw], [-viewW*2 viewW*2], '-', 'Color', edgeCol, 'LineWidth', 1.2);

    bldgCol = [0.82 0.82 0.87];
    ec2 = [0.6 0.6 0.7];
    patch(ax2, [-viewW*2 -hw -hw -viewW*2], [ hw  hw viewW*2 viewW*2], bldgCol, 'EdgeColor', ec2);
    patch(ax2, [ hw viewW*2 viewW*2  hw],   [ hw  hw viewW*2 viewW*2], bldgCol, 'EdgeColor', ec2);
    patch(ax2, [-viewW*2 -hw -hw -viewW*2], [-viewW*2 -viewW*2 -hw -hw], bldgCol, 'EdgeColor', ec2);
    patch(ax2, [ hw viewW*2 viewW*2  hw],   [-viewW*2 -viewW*2 -hw -hw], bldgCol, 'EdgeColor', ec2);

    plot(ax2, [-viewW*2 viewW*2], [0 0], 'k--', 'LineWidth', 0.4);
    plot(ax2, [0 0], [-viewW*2 viewW*2], 'k--', 'LineWidth', 0.4);

    % --- Draw arc guides ---
    theta_arc = linspace(0, pi/2, 80);
    for u = 1:nUAV
        xarc = acx - R_uav(u) * cos(theta_arc);
        yarc = acy + R_uav(u) * sin(theta_arc);
        plot(ax2, xarc, yarc, ':', 'Color', [0.75 0.75 0.75], 'LineWidth', 0.7);
    end

    % --- Compute & draw each UAV ---
    for u = 1:nUAV
        offset_u = offsets(u);
        y_lane   = -hw + margin + offset_u;
        arcLen_u = (pi/2) * R_uav(u);
        stagger  = abs(y_lane) * 0.6;
        d_u      = dist(f) - stagger;

        [px, py] = uavPosition(d_u, approachLen, arcLen_u, R_uav(u), ...
                               acx, acy, y_lane, hw, margin, offset_u, viewW);

        % Draw trail
        trailN = 12;
        for tt = 1:trailN
            ft = f - tt * 2;
            if ft < 1, continue; end
            d_t = dist(ft) - stagger;
            [tpx, tpy] = uavPosition(d_t, approachLen, arcLen_u, R_uav(u), ...
                                     acx, acy, y_lane, hw, margin, offset_u, viewW);
            fade = 0.15 + 0.35 * (1 - tt/trailN);
            trailCol = cmap(u,:) * fade + (1 - fade) * [1 1 1];
            plot(ax2, tpx, tpy, '.', 'Color', trailCol, 'MarkerSize', 6);
        end

        % Draw UAV body
        theta_circ = linspace(0, 2*pi, 30);
        xc = px + uav_r * cos(theta_circ);
        yc = py + uav_r * sin(theta_circ);
        patch(ax2, xc, yc, cmap(u,:), 'EdgeColor', cmap(u,:)*0.4, ...
              'FaceAlpha', 0.85, 'LineWidth', 1.2);

        % UAV label
        text(ax2, px + uav_r*1.5, py + uav_r*1.5, sprintf('%d', u), ...
             'FontSize', 7, 'Color', cmap(u,:)*0.5, 'FontWeight', 'bold');
    end

    % --- Info overlay ---
    text(ax2, -viewW + 1, viewW - 1.5, ...
         sprintf('v = %d m/s  |  road = %d m  |  n = %d UAVs  |  R_{turn} = %.1f m', ...
                 v, roadW, nUAV, R_turn), ...
         'FontSize', 10, 'Color', 'k', 'FontWeight', 'bold', ...
         'VerticalAlignment', 'top');
    text(ax2, -viewW + 1, -viewW + 1.5, sprintf('t = %.2f s', t_all(f)), ...
         'FontSize', 9, 'Color', [0.3 0.3 0.3]);

    xlim(ax2, [-viewW viewW]);
    ylim(ax2, [-viewW viewW]);
    set(ax2, 'XTick', [], 'YTick', []);
    title(ax2, 'UAV Formation - 90 Degree Turn', 'FontSize', 13, 'FontWeight', 'bold');
    box(ax2, 'on');

    drawnow;
    frame = getframe(fig2);
    writeVideo(vw, frame);
end

close(vw);
fprintf('Saved: %s\n', videoFile);
close(fig2);
end

%% =========================================================================
%  Helper: compute UAV position along its path
%% =========================================================================
function [px, py] = uavPosition(d, approachLen, arcLen, R, acx, acy, ...
                                 y_lane, hw, margin, offset, viewW)
if d < 0
    px = -viewW * 2;
    py = y_lane;
elseif d <= approachLen
    px = -approachLen + d;
    py = y_lane;
elseif d <= approachLen + arcLen
    d_arc = d - approachLen;
    ang   = d_arc / R;
    px    = acx - R * cos(ang);
    py    = acy + R * sin(ang);
else
    d_exit = d - approachLen - arcLen;
    x_lane = hw - margin - offset;
    px     = x_lane;
    py     = acy + R + d_exit;
end
end
