function drawEnvironment(hAx, env)
% Draw the intersection roads, buildings, and obstacles on axes handle hAx.
%
%   drawEnvironment(hAx, env)

w    = env.roadHalfW;
Xmin = env.bounds(1); Xmax = env.bounds(2);
Ymin = env.bounds(3); Ymax = env.bounds(4);

% Horizontal road
road1 = [Xmin, -w; Xmax, -w; Xmax, w; Xmin, w];
patch(hAx, road1(:,1), road1(:,2), [0.95 0.95 0.95], 'EdgeColor', 'none');

% Vertical road
road2 = [-w, Ymin; w, Ymin; w, Ymax; -w, Ymax];
patch(hAx, road2(:,1), road2(:,2), [0.95 0.95 0.95], 'EdgeColor', 'none');

% Buildings
for i = 1:numel(env.buildings)
    b = env.buildings{i};
    patch(hAx, b(:,1), b(:,2), [0.8 0.8 0.85], ...
        'EdgeColor', [0.6 0.6 0.7], 'LineWidth', 1.0);
end

% Obstacles (drawn in a distinct color — orange/red)
for i = 1:numel(env.obstacles)
    ob = env.obstacles{i};
    patch(hAx, ob(:,1), ob(:,2), [0.9 0.45 0.15], ...
        'EdgeColor', [0.7 0.2 0.0], 'LineWidth', 1.5, 'FaceAlpha', 0.85);
end

% Centerlines
plot(hAx, [Xmin Xmax], [0 0], 'k--', 'LineWidth', 0.75);
plot(hAx, [0 0], [Ymin Ymax], 'k--', 'LineWidth', 0.75);
end
