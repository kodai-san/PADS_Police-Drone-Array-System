function env2 = inflateBuildings(env, r)
% Inflate building polygons outward by radius r (preserving shape).
%
%   env2 = inflateBuildings(env, r)
%
% Each vertex is pushed outward from the polygon centroid by r meters.
% This preserves the polygon shape (unlike axis-aligned bounding boxes,
% which can cover roads for irregular footprints).
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
    cx = mean(b(:,1));
    cy = mean(b(:,2));
    % Push each vertex outward from centroid
    dx = b(:,1) - cx;
    dy = b(:,2) - cy;
    dist = sqrt(dx.^2 + dy.^2) + 1e-12;
    b_inf = b;
    b_inf(:,1) = b(:,1) + r * (dx ./ dist);
    b_inf(:,2) = b(:,2) + r * (dy ./ dist);
    env2.buildings{i} = b_inf;
end
end
