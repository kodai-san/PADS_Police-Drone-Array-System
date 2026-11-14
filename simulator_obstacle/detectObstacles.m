function [detected, detectedPolys, detectedCenters] = detectObstacles(dronePos, obstacles, detectRange)
% Simulate obstacle detection with a finite-range sensor.
%
%   [detected, detectedPolys, detectedCenters] = detectObstacles(dronePos, obstacles, detectRange)
%
% The sensor detects an obstacle if the drone is within detectRange of the
% nearest point on the obstacle polygon. This models a proximity sensor
% (lidar, ultrasonic, depth camera) rather than omniscient knowledge.
%
% Inputs:
%   dronePos    - [2x1] drone position [px; py]
%   obstacles   - cell array of Nx2 obstacle polygons
%   detectRange - detection radius (m)
%
% Outputs:
%   detected        - logical array, true for each detected obstacle
%   detectedPolys   - cell array of detected obstacle polygons only
%   detectedCenters - [2 x nDetected] centers of detected obstacles

nObs = numel(obstacles);
detected = false(1, nObs);
detectedPolys = {};
detectedCenters = zeros(2, 0);

dp = dronePos(:)';

for i = 1:nObs
    poly = obstacles{i};
    
    % Compute minimum distance from drone to any edge of the polygon
    dMin = minDistToPolygon(dp, poly);
    
    % Also check if drone is inside the polygon (distance = 0)
    if pointInPolyLocal(dp, poly)
        dMin = 0;
    end
    
    if dMin <= detectRange
        detected(i) = true;
        detectedPolys{end+1} = poly; %#ok<AGROW>
        % Compute polygon centroid
        cx = mean(poly(:,1));
        cy = mean(poly(:,2));
        detectedCenters(:, end+1) = [cx; cy]; %#ok<AGROW>
    end
end
end

%% ---- Local helpers ----

function dMin = minDistToPolygon(pt, poly)
% Minimum distance from point pt [1x2] to the edges of polygon poly [Nx2].
n = size(poly, 1);
dMin = inf;
for k = 1:n
    a = poly(k, :);
    b = poly(mod(k, n) + 1, :);
    d = pointSegmentDist(pt, a, b);
    if d < dMin
        dMin = d;
    end
end
end

function d = pointSegmentDist(p, a, b)
% Distance from point p to line segment a-b (all [1x2]).
ab = b - a;
ap = p - a;
t = dot(ap, ab) / (dot(ab, ab) + 1e-12);
t = max(0, min(1, t));
proj = a + t * ab;
d = norm(p - proj);
end

function tf = pointInPolyLocal(p, poly)
% Ray-casting point-in-polygon (local copy to keep detectObstacles self-contained).
x = p(1); y = p(2);
n = size(poly, 1);
inside = false;
j = n;
for i = 1:n
    xi = poly(i,1); yi = poly(i,2);
    xj = poly(j,1); yj = poly(j,2);
    crosses = ((yi > y) ~= (yj > y)) && ...
              (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi);
    if crosses
        inside = ~inside;
    end
    j = i;
end
tf = inside;
end
