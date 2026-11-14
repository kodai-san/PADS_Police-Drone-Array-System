function [t_vec, pos_vec] = build_target(waypoints, speed, T_max)
% BUILD_TARGET  Build time-parameterised target trajectory at constant speed.
%
%   [t_vec, pos_vec] = build_target(waypoints, speed, T_max)
%
%   waypoints — Nx2 array of [x,y] positions
%   speed     — constant speed along the path [m/s]
%   T_max     — maximum time to precompute [s]
%
%   t_vec   — 1xM time vector
%   pos_vec — 2xM position matrix
%
%   After reaching the last waypoint the target continues in the
%   direction of the final segment at the same speed.

    n_wp = size(waypoints, 1);

    % Cumulative arc-length at each waypoint
    cum_dist = zeros(n_wp, 1);
    for i = 2:n_wp
        cum_dist(i) = cum_dist(i-1) + norm(waypoints(i,:) - waypoints(i-1,:));
    end
    total_dist = cum_dist(end);

    % Fine time grid
    dt_fine = 0.01;
    t_vec   = 0 : dt_fine : T_max;
    pos_vec = zeros(2, length(t_vec));

    for i = 1:length(t_vec)
        d = speed * t_vec(i);

        if d >= total_dist
            % Past last waypoint — extrapolate along final direction
            dir = waypoints(end,:) - waypoints(end-1,:);
            dir = dir / norm(dir);
            pos_vec(:,i) = waypoints(end,:)' + dir' * (d - total_dist);
        else
            % Find active segment
            seg  = find(cum_dist(2:end) >= d, 1);
            frac = (d - cum_dist(seg)) / (cum_dist(seg+1) - cum_dist(seg));
            pos_vec(:,i) = ((1-frac)*waypoints(seg,:) + frac*waypoints(seg+1,:))';
        end
    end
end
