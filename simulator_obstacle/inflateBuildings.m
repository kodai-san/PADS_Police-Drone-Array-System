function env2 = inflateBuildings(env, r)
% Inflate axis-aligned rectangular buildings and obstacles by radius r.
%
%   env2 = inflateBuildings(env, r)
%
% Inputs:
%   env - environment struct with .buildings and .obstacles cell arrays
%   r   - inflation radius (m)
%
% Outputs:
%   env2 - copy of env with inflated building and obstacle polygons,
%          and updated .allSolids

env2 = env;

% Inflate buildings
for i = 1:numel(env.buildings)
    env2.buildings{i} = inflateRect(env.buildings{i}, r);
end

% Inflate obstacles
if isfield(env, 'obstacles')
    for i = 1:numel(env.obstacles)
        env2.obstacles{i} = inflateRect(env.obstacles{i}, r);
    end
else
    env2.obstacles = {};
end

% Rebuild combined list
env2.allSolids = [env2.buildings, env2.obstacles];
end

function poly = inflateRect(poly, r)
% Inflate an axis-aligned rectangle polygon by r on all sides.
xmin = min(poly(:,1)) - r;
xmax = max(poly(:,1)) + r;
ymin = min(poly(:,2)) - r;
ymax = max(poly(:,2)) + r;
poly = [xmin, ymin; xmax, ymin; xmax, ymax; xmin, ymax];
end
