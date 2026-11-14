function env = makeIntersectionEnvironment(roadW, obstacles)
% Create an intersection environment with four building blocks and optional
% road obstacles.
%
%   env = makeIntersectionEnvironment(roadW)
%   env = makeIntersectionEnvironment(roadW, obstacles)
%
% Inputs:
%   roadW     - total road width (m)
%   obstacles - (optional) cell array of obstacle structs, each with fields:
%                 .center  [x; y]   - center position
%                 .size    [w; h]   - width and height (m)
%               If omitted, no obstacles are placed.
%
% Outputs:
%   env - struct with fields:
%           .buildings  - cell array of Nx2 building polygons
%           .obstacles  - cell array of Nx2 obstacle polygons (separate from buildings)
%           .allSolids  - cell array of ALL solid polygons (buildings + obstacles)
%           .roadHalfW  - half road width
%           .bounds     - [Xmin Xmax Ymin Ymax]

if nargin < 2
    obstacles = {};
end

w = roadW / 2;
Xmin = -60; Xmax = 60;
Ymin = -60; Ymax = 60;

% Four quadrant buildings
b1 = [Xmin,  w;  -w,  w;  -w, Ymax;  Xmin, Ymax];  % upper-left
b2 = [  w,   w; Xmax,  w; Xmax, Ymax;    w, Ymax];  % upper-right
b3 = [Xmin, Ymin;  -w, Ymin;  -w, -w;  Xmin, -w];   % lower-left
b4 = [  w,  Ymin; Xmax, Ymin; Xmax, -w;    w, -w];   % lower-right

env.buildings = {b1, b2, b3, b4};
env.roadHalfW = w;
env.bounds    = [Xmin, Xmax, Ymin, Ymax];

% Convert obstacle structs to polygons
obsPoly = cell(1, numel(obstacles));
for i = 1:numel(obstacles)
    obs = obstacles{i};
    cx = obs.center(1);
    cy = obs.center(2);
    hw = obs.size(1) / 2;
    hh = obs.size(2) / 2;
    obsPoly{i} = [cx-hw, cy-hh;
                   cx+hw, cy-hh;
                   cx+hw, cy+hh;
                   cx-hw, cy+hh];
end
env.obstacles = obsPoly;

% Combined solid geometry (buildings + obstacles)
env.allSolids = [env.buildings, env.obstacles];
end
