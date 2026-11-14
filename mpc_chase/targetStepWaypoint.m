function [xnext, wpIdx] = targetStepWaypoint(x, wpIdx, pathPts, speed, dt)
% Advance target one time step along a sequence of path points at constant speed.
%
%   [xnext, wpIdx] = targetStepWaypoint(x, wpIdx, pathPts, speed, dt)
%
% The target moves toward the next path point. When it arrives within one
% step's travel distance, it advances to the following point. When the end
% of the path is reached, the target stops.
%
% Inputs:
%   x       - [4x1] target state [px; py; vx; vy]
%   wpIdx   - current index into pathPts (next point to head toward)
%   pathPts - [P x 2] ordered path points (meters, ENU)
%   speed   - target walking speed (m/s)
%   dt      - time step (s)
%
% Outputs:
%   xnext - [4x1] updated target state
%   wpIdx - updated waypoint index

nPts = size(pathPts, 1);
pos  = x(1:2);

% If we've reached the end of the path, stop
if wpIdx > nPts
    xnext = x;
    xnext(3:4) = [0; 0];
    return;
end

% Distance to next waypoint
goal = pathPts(wpIdx, :)';
toGoal = goal - pos;
distToGoal = norm(toGoal);

stepDist = speed * dt;

% Advance through waypoints if we can overshoot in this step
while distToGoal < stepDist && wpIdx < nPts
    % Consume distance to reach this waypoint
    stepDist = stepDist - distToGoal;
    pos = goal;

    wpIdx = wpIdx + 1;
    if wpIdx > nPts
        break;
    end
    goal = pathPts(wpIdx, :)';
    toGoal = goal - pos;
    distToGoal = norm(toGoal);
end

% Move remaining distance toward current goal
if wpIdx <= nPts && distToGoal > 1e-6
    dir = toGoal / distToGoal;
    pos = pos + dir * stepDist;
    vel = dir * speed;
else
    vel = [0; 0];
end

xnext = [pos; vel];
end
