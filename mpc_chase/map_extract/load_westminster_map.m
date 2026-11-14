function mapData = load_westminster_map(matFile)
% Load the Westminster map data exported by extract_westminster_map.py
% and convert it into the format expected by the simulation.
%
%   mapData = load_westminster_map()                     % default file
%   mapData = load_westminster_map('westminster_map.mat')
%
% Output struct fields:
%   .road_segments  - [N x 4] road line segments [x1 y1 x2 y2] (meters, ENU)
%   .road_nodes     - [M x 2] road node positions (meters, ENU)
%   .buildings      - cell array of Kx2 building polygon vertices
%   .target_path    - [P x 2] ordered path points for target to follow
%   .waypoints      - [W x 2] high-level waypoints (ENU)
%   .ref_latlon     - [1x2] reference lat/lon for ENU origin
%   .bounds         - [xmin xmax ymin ymax] bounding box in ENU

if nargin < 1
    matFile = 'westminster_map.mat';
end

fprintf('Loading map from: %s\n', matFile);
S = load(matFile);

mapData.road_segments = S.road_segments;
mapData.road_nodes    = S.road_nodes;
mapData.target_path   = S.target_path;
mapData.waypoints     = S.waypoints_enu;
mapData.ref_latlon    = S.ref_latlon;

% Convert building polygons from cell/object array
nBldg = double(S.n_buildings);
mapData.buildings = cell(1, nBldg);
for i = 1:nBldg
    mapData.buildings{i} = double(S.building_polys{i});
end

% Compute bounds
allX = [mapData.road_nodes(:,1); mapData.target_path(:,1)];
allY = [mapData.road_nodes(:,2); mapData.target_path(:,2)];
margin = 50;
mapData.bounds = [min(allX)-margin, max(allX)+margin, ...
                  min(allY)-margin, max(allY)+margin];

fprintf('  Roads:     %d segments, %d nodes\n', ...
    size(mapData.road_segments,1), size(mapData.road_nodes,1));
fprintf('  Buildings: %d polygons\n', nBldg);
fprintf('  Target path: %d points\n', size(mapData.target_path,1));
fprintf('  Bounds: E [%.0f, %.0f] m, N [%.0f, %.0f] m\n', ...
    mapData.bounds(1), mapData.bounds(2), mapData.bounds(3), mapData.bounds(4));
end
