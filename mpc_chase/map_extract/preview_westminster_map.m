function preview_westminster_map(matFile)
% Preview the extracted Westminster map data.
% Draws buildings, roads, target path, and waypoints for visual verification.
%
%   preview_westminster_map()
%   preview_westminster_map('westminster_map.mat')

if nargin < 1
    matFile = 'westminster_map.mat';
end

mapData = load_westminster_map(matFile);

figure('Color','w','Name','Westminster Map Preview','Position',[100 100 900 700]);
hAx = axes; hold(hAx,'on'); axis(hAx,'equal'); grid(hAx,'on');
xlabel(hAx,'East (m)'); ylabel(hAx,'North (m)');
title(hAx,'Westminster Map — OSM Extract');

% Draw roads (light grey lines)
segs = mapData.road_segments;
for i = 1:size(segs,1)
    plot(hAx, [segs(i,1) segs(i,3)], [segs(i,2) segs(i,4)], ...
        '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 0.8);
end

% Draw buildings (blue-grey fill)
for i = 1:numel(mapData.buildings)
    b = mapData.buildings{i};
    if size(b,1) >= 3
        patch(hAx, b(:,1), b(:,2), [0.75 0.78 0.85], ...
            'EdgeColor', [0.5 0.55 0.65], 'LineWidth', 0.5, 'FaceAlpha', 0.7);
    end
end

% Draw target path (red line)
tp = mapData.target_path;
plot(hAx, tp(:,1), tp(:,2), 'r-', 'LineWidth', 2.0);
plot(hAx, tp(1,1), tp(1,2), 'go', 'MarkerSize', 12, 'LineWidth', 2); % start
plot(hAx, tp(end,1), tp(end,2), 'rx', 'MarkerSize', 12, 'LineWidth', 2); % end

% Draw waypoints (magenta diamonds)
wp = mapData.waypoints;
plot(hAx, wp(:,1), wp(:,2), 'md', 'MarkerSize', 10, 'LineWidth', 1.5, ...
    'MarkerFaceColor', [0.8 0.2 0.8]);
for i = 1:size(wp,1)
    text(hAx, wp(i,1)+5, wp(i,2)+5, sprintf('WP%d', i), ...
        'FontSize', 8, 'Color', [0.6 0.1 0.6]);
end

% Set axis limits
xlim(hAx, [mapData.bounds(1), mapData.bounds(2)]);
ylim(hAx, [mapData.bounds(3), mapData.bounds(4)]);

legend(hAx, {'Roads','','Target path','Start','End','Waypoints'}, ...
    'Location','best');

% Print path stats
pathLen = sum(sqrt(sum(diff(tp).^2, 2)));
fprintf('Target path total length: %.0f m\n', pathLen);
fprintf('Number of waypoints: %d\n', size(wp,1));
fprintf('Number of path nodes: %d\n', size(tp,1));
end
