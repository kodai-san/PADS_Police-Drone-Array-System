function env2 = inflateBuildings(env, r)
% Inflate axis-aligned rectangular buildings by radius r.
%
%   env2 = inflateBuildings(env, r)
%
% Inputs:
%   env - environment struct with .buildings cell array of Nx2 polygons
%   r   - inflation radius (m)
%
% Outputs:
%   env2 - copy of env with inflated building polygons

env2 = env;
for i = 1:numel(env.buildings)
    b = env.buildings{i};
    xmin = min(b(:,1)) - r;
    xmax = max(b(:,1)) + r;
    ymin = min(b(:,2)) - r;
    ymax = max(b(:,2)) + r;
    % Counter-clockwise winding for consistency
    env2.buildings{i} = [xmin, ymin; xmax, ymin; xmax, ymax; xmin, ymax];
end
end
