function a_repulse = obstacleRepulsion(dronePos, droneVel, detectedPolys, detectedCenters, P)
% Compute a repulsive acceleration vector to steer away from detected obstacles.
%
%   a_repulse = obstacleRepulsion(dronePos, droneVel, detectedPolys, detectedCenters, P)
%
% Uses an artificial potential field approach:
%   - Repulsive force is inversely proportional to distance squared
%   - Direction is away from the nearest point on the obstacle
%   - Magnitude is scaled by P.obsRepulseGain and capped at P.obsRepulseMax
%   - A tangential component is added to encourage going around rather than
%     just pushing back (avoids local minima / oscillation)
%
% Inputs:
%   dronePos        - [2x1] drone position
%   droneVel        - [2x1] drone velocity
%   detectedPolys   - cell array of detected obstacle polygons
%   detectedCenters - [2 x nDetected] obstacle centers
%   P               - parameter struct with fields:
%                       .obsDetectRange, .obsRepulseGain, .obsRepulseMax
%
% Outputs:
%   a_repulse - [2x1] repulsive acceleration to add to the MPC command

a_repulse = [0; 0];
dp = dronePos(:)';

for i = 1:numel(detectedPolys)
    poly = detectedPolys{i};
    
    % Find nearest point on polygon boundary
    [nearPt, dMin] = nearestPointOnPolygon(dp, poly);
    
    if dMin < 1e-3
        dMin = 1e-3;  % avoid division by zero
    end
    
    % Repulsive direction: away from nearest point
    awayDir = (dronePos(:) - nearPt(:));
    awayNorm = norm(awayDir);
    if awayNorm < 1e-6
        % Exactly on the obstacle — push away from center
        awayDir = dronePos(:) - detectedCenters(:, i);
        awayNorm = norm(awayDir);
        if awayNorm < 1e-6
            awayDir = [1; 0]; % fallback
            awayNorm = 1;
        end
    end
    awayDir = awayDir / awayNorm;
    
    % Repulsive magnitude: stronger when closer
    %   F = gain * (1/d - 1/d_max) / d^2   (classic potential field)
    %   Only active when d < detectRange
    d_max = P.obsDetectRange;
    if dMin < d_max
        mag = P.obsRepulseGain * (1/dMin - 1/d_max) * (1/dMin^2);
        mag = min(mag, P.obsRepulseMax);
    else
        mag = 0;
    end
    
    % Radial repulsion
    a_radial = mag * awayDir;
    
    % Tangential component: cross product with velocity to encourage
    % going around the obstacle rather than just backing up
    speed = norm(droneVel);
    if speed > 0.3
        vhat = droneVel(:) / speed;
        % Choose tangent direction that has positive dot with away direction
        tang1 = [-vhat(2);  vhat(1)];
        tang2 = [ vhat(2); -vhat(1)];
        if dot(tang1, awayDir) > dot(tang2, awayDir)
            tangDir = tang1;
        else
            tangDir = tang2;
        end
        a_tang = 0.5 * mag * tangDir;
    else
        a_tang = [0; 0];
    end
    
    a_repulse = a_repulse + a_radial + a_tang;
end

% Cap total repulsive acceleration
if norm(a_repulse) > P.obsRepulseMax
    a_repulse = a_repulse / norm(a_repulse) * P.obsRepulseMax;
end
end

%% ---- Local helper ----

function [nearPt, dMin] = nearestPointOnPolygon(pt, poly)
% Find the nearest point on the boundary of polygon poly to point pt [1x2].
n = size(poly, 1);
dMin = inf;
nearPt = poly(1, :)';

for k = 1:n
    a = poly(k, :);
    b = poly(mod(k, n) + 1, :);
    [proj, d] = projectPointToSegment(pt, a, b);
    if d < dMin
        dMin = d;
        nearPt = proj(:);
    end
end
end

function [proj, d] = projectPointToSegment(p, a, b)
% Project point p onto segment a-b and return closest point + distance.
ab = b - a;
ap = p - a;
t = dot(ap, ab) / (dot(ab, ab) + 1e-12);
t = max(0, min(1, t));
proj = a + t * ab;
d = norm(p - proj);
end
