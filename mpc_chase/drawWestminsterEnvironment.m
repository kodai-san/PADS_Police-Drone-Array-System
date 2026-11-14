function drawWestminsterEnvironment(hAx, mapData, targetPath)
% Draw the Westminster map on the given axes.
%
%   drawWestminsterEnvironment(hAx, mapData)
%   drawWestminsterEnvironment(hAx, mapData, targetPath)
%
% Inputs:
%   hAx        - axes handle
%   mapData    - struct from load_westminster_map
%   targetPath - (optional) [Px2] path to draw

% Roads
segs = mapData.road_segments;
for i = 1:size(segs,1)
    plot(hAx, [segs(i,1) segs(i,3)], [segs(i,2) segs(i,4)], ...
        '-', 'Color', [0.88 0.88 0.88], 'LineWidth', 0.6);
end

% Buildings
for i = 1:numel(mapData.buildings)
    b = mapData.buildings{i};
    if size(b,1) >= 3
        patch(hAx, b(:,1), b(:,2), [0.78 0.80 0.87], ...
            'EdgeColor', [0.55 0.58 0.68], 'LineWidth', 0.4, 'FaceAlpha', 0.75);
    end
end

% Target path (thin red)
if nargin >= 3 && ~isempty(targetPath)
    plot(hAx, targetPath(:,1), targetPath(:,2), 'r-', 'LineWidth', 1.0, ...
        'Color', [1 0.3 0.3 0.5]);
end
end
