%% UAV Formation Fitting in Road + Safe 90 deg Turn (MATLAB)
% What this does:
% 1) Builds a 90 degree road corridor (width can change before and after corner)
% 2) Picks a formation that fits the minimum road width given speed
% 3) Generates collision-safe trajectories through the corner using arc turns
% 4) Simulates and animates UAVs, checks minimum separation
%
% Notes:
% - Simple planar kinematics, constant speed, coordinated turn constraint via max lateral accel
% - Collision avoidance is ensured by design using (a) longitudinal time headway and
%   (b) formation offsets that fit the corridor with margin.

clear; clc; close all;

%% User inputs
N = 6;                   % number of UAVs
v = 12;                  % speed (m/s)
aLatMax = 4.0;           % max lateral acceleration (m/s^2) for coordinated turn
minSep = 3.0;            % required minimum inter-UAV separation (m)
tHeadway = 0.35;         % seconds of time headway along path (for collision avoidance)
simDt = 0.05;            % simulation timestep (s)

% Road widths (meters)
W_before = 8;            % road width on incoming segment
W_after  = 6;            % road width on outgoing segment
W_corner = 6;            % effective width near corner (tightest)
W_min = min([W_before, W_after, W_corner]);

% Corridor geometry
L_in  = 60;              % length of incoming straight (m)
L_out = 60;              % length of outgoing straight (m)

% Corner point at (0,0). Incoming along +x to the corner, then turn left to +y.
% Incoming centerline: y=0, x from -L_in to 0
% Outgoing centerline: x=0, y from 0 to L_out

%% Derived constraints
Rmin = v^2 / aLatMax;           % minimum turn radius (m)
Rturn = max(Rmin, 8);           % choose a feasible radius, clamp for visual stability

% Safety margins
marginWall = 0.8;               % distance from UAV center to road boundary (m)
maxLatAllow = (W_min/2) - marginWall;     % max allowed lateral offset magnitude from centerline
if maxLatAllow <= 0
    error('Road too narrow after applying wall margin.');
end

%% Define candidate formations
% Each formation is defined by lateral offsets (across-road) for each UAV at any time.
% Longitudinal separation is provided by time headway.
%
% You can add formations by defining function handles that return offsets.
formations = getFormationLibrary(N, minSep, maxLatAllow);

%% Choose a formation that fits width and is reasonable for speed
% Speed can tighten constraints because higher speed implies larger turn radius,
% which increases required "sweep" space. We conservatively reduce lateral allowance
% when Rturn is close to corridor scale.
%
% Simple conservatism factor: reduce max lateral allowance if turn radius is big
% relative to corner width. This is a heuristic, not a full vehicle dynamics model.
turnTightness = min(1.0, (W_min) / max(1e-6, Rturn));   % smaller means tighter
latAllowEffective = maxLatAllow * (0.65 + 0.35*turnTightness);

[formationName, latOffsets] = pickFormation(formations, latAllowEffective);

fprintf('Selected formation: %s\n', formationName);
fprintf('Speed = %.2f m/s, Rmin = %.2f m, Rturn = %.2f m\n', v, Rmin, Rturn);
fprintf('W_min = %.2f m, maxLatAllow = %.2f m, effective = %.2f m\n', ...
    W_min, maxLatAllow, latAllowEffective);

%% Build a reference path for the formation leader (centerline)
% Path: straight in (x), then quarter circle of radius Rturn, then straight out (y).
[path, sTotal] = build90degPath(L_in, L_out, Rturn);

% Convert speed to path time
T_total = sTotal / v;
t = 0:simDt:T_total;

% Leader arc-length over time
sLeader = v * t;

%% Generate trajectories for all UAVs
% Each UAV i follows the same path but shifted in arc-length (time headway)
% and shifted laterally by its formation offset.
% Lateral shift is applied in the path's normal direction.

% Longitudinal staggering (path distance)
% Put UAV 1 as front leader, UAV 2 behind, etc.
sShift = (0:N-1) * (v * tHeadway);  % meters behind the leader

% Allocate
pos = zeros(numel(t), N, 2);   % [time, uav, (x,y)]
yaw = zeros(numel(t), N);      % heading angle (rad)

for k = 1:numel(t)
    for i = 1:N
        s_i = max(0, sLeader(k) - sShift(i));  % lag behind leader
        [p, psi, nHat] = samplePath(path, s_i);

        % Lateral offset for formation (clipped by effective allowance)
        d = clamp(latOffsets(i), -latAllowEffective, latAllowEffective);

        % Apply lateral offset in normal direction
        p_i = p + d * nHat';

        pos(k,i,1:2) = p_i;
        yaw(k,i) = psi;
    end
end

%% Collision check (pairwise distances)
[minDist, minDistTime, pair] = computeMinSeparation(pos);

fprintf('Minimum pairwise separation = %.3f m at t = %.2f s between UAV %d and %d\n', ...
    minDist, t(minDistTime), pair(1), pair(2));

if minDist < minSep
    warning('Min separation violated. Increase tHeadway or minSep, or reduce lateral offsets.');
end

%% Road corridor drawing and animation
figure('Color','w'); hold on; axis equal; grid on;
title(sprintf('UAV formation "%s" through 90 deg turn (v=%.1f m/s)', formationName, v));
xlabel('x (m)'); ylabel('y (m)');

drawRoadCorridor(L_in, L_out, W_before, W_after, W_corner);

% Plot path centerline
plot(path.xy(:,1), path.xy(:,2), 'k--', 'LineWidth', 1);

% UAV markers
colors = lines(N);
h = gobjects(N,1);
for i = 1:N
    h(i) = plot(nan, nan, 'o', 'MarkerSize', 8, 'LineWidth', 2, ...
        'Color', colors(i,:), 'MarkerFaceColor', colors(i,:));
end

% Optional: draw separation circles
hCirc = gobjects(N,1);
theta = linspace(0, 2*pi, 60);
for i = 1:N
    hCirc(i) = plot(nan, nan, '-', 'Color', colors(i,:), 'LineWidth', 1);
end

% Animate
for k = 1:4:numel(t)
    for i = 1:N
        x = pos(k,i,1); y = pos(k,i,2);
        set(h(i), 'XData', x, 'YData', y);

        cx = x + minSep*cos(theta);
        cy = y + minSep*sin(theta);
        set(hCirc(i), 'XData', cx, 'YData', cy);
    end
    drawnow;
end

%% --------- Functions ---------

function formations = getFormationLibrary(N, minSep, maxLatAllow)
% Returns a struct array of candidate formations.
% Each entry has:
% - name
% - offsets: Nx1 lateral offsets (meters) relative to road centerline

formations = struct('name', {}, 'offsets', {});

% 1) Column (single file)
offsets = zeros(N,1);
formations(end+1) = struct('name','column', 'offsets', offsets);

% 2) Line abreast (spread across road)
% spread symmetrically with spacing = minSep
offsets = spreadSymmetric(N, minSep);
formations(end+1) = struct('name','line_abreast', 'offsets', offsets);

% 3) Staggered (two columns)
% alternating left/right at half spacing
offsets = zeros(N,1);
d = minSep/2;
for i = 1:N
    offsets(i) = ((-1)^(i)) * d;
end
formations(end+1) = struct('name','staggered_two', 'offsets', offsets);

% 4) V formation
% leader center, then pairs outward
offsets = zeros(N,1);
step = minSep/2;
idx = 2;
level = 1;
while idx <= N
    if idx <= N, offsets(idx) =  level*step; idx = idx+1; end
    if idx <= N, offsets(idx) = -level*step; idx = idx+1; end
    level = level + 1;
end
formations(end+1) = struct('name','V', 'offsets', offsets);

% Clip all formations by absolute allowance to avoid silly widths
for k = 1:numel(formations)
    formations(k).offsets = clamp(formations(k).offsets, -maxLatAllow, maxLatAllow);
end
end

function [name, offsets] = pickFormation(formations, latAllow)
% Select formation that fits within latAllow and maximizes lateral spread modestly,
% but prefers narrower at higher constraint.
bestScore = -inf;
bestIdx = 1;

for k = 1:numel(formations)
    off = formations(k).offsets;
    if max(abs(off)) <= latAllow
        width = max(off) - min(off);
        % Heuristic score: prefer fitting with some spread, but not huge.
        score = 1.0*width - 0.15*(width^2);
        if score > bestScore
            bestScore = score;
            bestIdx = k;
        end
    end
end

name = formations(bestIdx).name;
offsets = formations(bestIdx).offsets;
end

function [path, sTotal] = build90degPath(L_in, L_out, R)
% Build polyline samples + per-s cumulative s and headings.
% Incoming straight: from x=-L_in to x=0 at y=0
% Turn: quarter circle centered at (-R, R), from angle -pi/2 to 0 (left turn)
% Outgoing straight: from (0,R) to (0, R+L_out) but shifted to match corner at (0,0)
%
% We want corner at (0,0). Use incoming straight to (0,0),
% then arc from (0,0) to (R,R) centered at (0,R). Actually easier:
% Place arc center at (0, R), start at (0,0), end at (R,R).
% That implies incoming along +x to (0,0) then left turn to +y at x=R end.
% But outgoing straight should start at (R,R) going +y. We then shift x by -R
% to keep outgoing on x=0. Instead, define:
% Incoming: x in [-L_in, 0], y=0
% Arc center at (0, R): start at (0,0) heading +x, end at (-R, R) heading +y.
% Then outgoing straight: x=-R, y in [R, R+L_out].
%
% That keeps a clean 90 deg left turn at the corner point (0,0) with arc radius R.
% Road centerline turns around the inside of corner.

n1 = 400; n2 = 300; n3 = 400;

% Segment 1
x1 = linspace(-L_in, 0, n1)';
y1 = zeros(n1,1);

% Segment 2: quarter circle centered at (0, R)
% Start angle = -pi/2 gives point (0,0). End angle = -pi gives point (-R, R).
ang = linspace(-pi/2, -pi, n2)';
x2 = 0 + R*cos(ang);
y2 = R + R*sin(ang);

% Segment 3
x3 = (-R) * ones(n3,1);
y3 = linspace(R, R+L_out, n3)';

xy = [x1 y1; x2 y2; x3 y3];

% Compute s, heading
ds = sqrt(sum(diff(xy,1,1).^2,2));
s = [0; cumsum(ds)];
psi = atan2([diff(xy(:,2)); 0], [diff(xy(:,1)); 0]);

path.xy = xy;
path.s = s;
path.psi = psi;

sTotal = s(end);
end

function [p, psi, nHat] = samplePath(path, sQuery)
% Sample a polyline path by arc-length, return position, heading, and normal.
s = path.s;
xy = path.xy;
psiArr = path.psi;

if sQuery <= 0
    p = xy(1,:);
    psi = psiArr(1);
else
    if sQuery >= s(end)
        p = xy(end,:);
        psi = psiArr(end);
    else
        idx = find(s <= sQuery, 1, 'last');
        if idx == numel(s)
            idx = numel(s)-1;
        end
        s0 = s(idx); s1 = s(idx+1);
        a = (sQuery - s0) / max(1e-9, (s1 - s0));
        p = (1-a)*xy(idx,:) + a*xy(idx+1,:);
        psi = (1-a)*psiArr(idx) + a*psiArr(idx+1);
    end
end

% Unit normal (left) from heading
tHat = [cos(psi), sin(psi)];
nHat = [-tHat(2), tHat(1)];
end

function [minDist, kMin, pair] = computeMinSeparation(pos)
% pos: [T, N, 2]
T = size(pos,1);
N = size(pos,2);
minDist = inf;
kMin = 1;
pair = [1 2];

for k = 1:T
    P = squeeze(pos(k,:,:)); % Nx2
    for i = 1:N-1
        for j = i+1:N
            d = norm(P(i,:) - P(j,:));
            if d < minDist
                minDist = d;
                kMin = k;
                pair = [i j];
            end
        end
    end
end
end

function drawRoadCorridor(L_in, L_out, Wb, Wa, Wc)
% Draw a simple corridor around the centerline with three widths.
% Incoming segment around y=0, x in [-L_in, 0] width Wb
% Corner region near x in [-Wc, 0], y in [0, Wc] width Wc (approx)
% Outgoing segment around x=-R? but we do not know R here, so we draw a generic
% outgoing around x=-Wc/2 as a visual aid. This is approximate for display.

% Incoming boundaries
xin = [-L_in, 0];
plot(xin, [ Wb/2,  Wb/2], 'k', 'LineWidth', 2);
plot(xin, [-Wb/2, -Wb/2], 'k', 'LineWidth', 2);

% Corner box approximation
xBox = [-Wc, 0, 0, -Wc, -Wc];
yBoxTop = [Wc/2, Wc/2, -Wc/2, -Wc/2, Wc/2];
plot(xBox, yBoxTop, 'k', 'LineWidth', 2);

% Outgoing boundaries approximation near x=-Wc
yout = [0, L_out];
plot([-Wc/2 + Wa/2, -Wc/2 + Wa/2], yout, 'k', 'LineWidth', 2);
plot([-Wc/2 - Wa/2, -Wc/2 - Wa/2], yout, 'k', 'LineWidth', 2);

end

function x = spreadSymmetric(N, spacing)
% Symmetric offsets centered at 0 with given spacing between neighbors.
% Example N=5 -> [-2 -1 0 1 2]*spacing
idx = (0:N-1)';
center = (N-1)/2;
x = (idx - center) * spacing;
end

function y = clamp(x, lo, hi)
y = min(max(x, lo), hi);
end