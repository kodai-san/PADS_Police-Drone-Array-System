function env = makeWestminsterEnvironment(mapData)
% Convert Westminster map data into the env struct format used by the
% simulation (compatible with hasLineOfSight, pointInAnyBuilding, etc.)
%
%   env = makeWestminsterEnvironment(mapData)
%
% Inputs:
%   mapData - struct from load_westminster_map
%
% Outputs:
%   env - struct with fields:
%           .buildings  - cell array of building polygons
%           .obstacles  - empty cell (no separate obstacles by default)
%           .allSolids  - same as buildings
%           .roadHalfW  - not applicable (set to 0)
%           .bounds     - [xmin xmax ymin ymax]

env.buildings = mapData.buildings;
env.obstacles = {};
env.allSolids = mapData.buildings;
env.roadHalfW = 0;  % not used in Westminster mode
env.bounds    = mapData.bounds;
end
