function [a_safe, used] = safetyFilterAccel(xd, a_cmd, A, B, envInf, P, detectedObsInf)
% Safety filter: find an acceleration close to a_cmd that keeps the
% drone outside inflated buildings AND detected obstacles at the next step.
%
%   [a_safe, used] = safetyFilterAccel(xd, a_cmd, A, B, envInf, P)
%   [a_safe, used] = safetyFilterAccel(xd, a_cmd, A, B, envInf, P, detectedObsInf)
%
% Inputs:
%   xd             - [4x1] current drone state
%   a_cmd          - [2x1] commanded acceleration
%   A, B           - discrete dynamics matrices
%   envInf         - inflated environment struct (buildings)
%   P              - parameter struct
%   detectedObsInf - (optional) cell array of inflated detected obstacle
%                    polygons to additionally avoid
%
% Outputs:
%   a_safe - [2x1] safe acceleration command
%   used   - true if the filter had to modify the command

if nargin < 7
    detectedObsInf = {};
end

% Combine all solids to check against
allPolys = [envInf.buildings, detectedObsInf];

used = false;

% Predict next state with commanded accel
x1 = A*xd + B*a_cmd;
if ~pointInAnyPoly(x1(1:2), allPolys)
    a_safe = a_cmd;
    return;
end

used = true;

% Build candidate accelerations
cand = zeros(2, 0);

% 1) Braking candidate (oppose current velocity)
if P.emergencyBrake
    v = xd(3:4);
    if norm(v) > 1e-3
        a_brake = -P.amax * (v / (norm(v) + 1e-12));
    else
        a_brake = [0; 0];
    end
    cand(:, end+1) = a_brake; %#ok<AGROW>
end

% 2) Rotate the command direction through uniformly spaced angles
a0 = a_cmd;
if norm(a0) < 1e-6
    a0 = [P.amax; 0];
end
a0 = a0 / (norm(a0) + 1e-12) * P.amax;

for i = 1:P.safetyCandidates
    ang = 2*pi * (i-1) / P.safetyCandidates;
    R = [cos(ang) -sin(ang); sin(ang) cos(ang)];
    cand(:, end+1) = R * a0; %#ok<AGROW>
end

% 3) Purely lateral relative to velocity (helps corridor following)
v = xd(3:4);
if norm(v) > 1e-3
    vhat = v / (norm(v) + 1e-12);
    lat1 = [-vhat(2);  vhat(1)] * P.amax;
    lat2 = [ vhat(2); -vhat(1)] * P.amax;
    cand(:, end+1) = lat1; %#ok<AGROW>
    cand(:, end+1) = lat2; %#ok<AGROW>
end

% 4) Reduced-magnitude candidates (half thrust in all directions)
for i = 1:P.safetyCandidates
    ang = 2*pi * (i-1) / P.safetyCandidates;
    R = [cos(ang) -sin(ang); sin(ang) cos(ang)];
    cand(:, end+1) = R * a0 * 0.5; %#ok<AGROW>
end

% 5) Zero acceleration
cand(:, end+1) = [0; 0]; %#ok<AGROW>

% Evaluate candidates: pick the one closest to a_cmd that is safe
best = [];
bestCost = inf;
for j = 1:size(cand, 2)
    a = cand(:, j);
    x1 = A*xd + B*a;
    if pointInAnyPoly(x1(1:2), allPolys)
        continue;
    end
    cost = norm(a - a_cmd);
    if cost < bestCost
        bestCost = cost;
        best = a;
    end
end

if isempty(best)
    % No safe candidate found: emergency stop
    a_safe = [0; 0];
else
    a_safe = best;
end
end

%% ---- Local helper ----
function tf = pointInAnyPoly(p, polys)
tf = false;
for i = 1:numel(polys)
    if pointInPoly(p, polys{i})
        tf = true;
        return;
    end
end
end
