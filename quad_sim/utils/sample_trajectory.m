function [ref_pos, ref_vel] = sample_trajectory(t, traj)
% SAMPLE_TRAJECTORY  Interpolate planned trajectory at time t.
%
%   [ref_pos, ref_vel] = sample_trajectory(t, traj)
%
%   traj — Nx5 matrix: [t, x, y, vx, vy]

    if isempty(traj)
        ref_pos = [0; 0];
        ref_vel = [0; 0];
        return;
    end

    t_traj = traj(:, 1);

    if t <= t_traj(1)
        ref_pos = traj(1, 2:3)';
        ref_vel = traj(1, 4:5)';
    elseif t >= t_traj(end)
        dt      = t - t_traj(end);
        ref_vel = traj(end, 4:5)';
        ref_pos = traj(end, 2:3)' + ref_vel * dt;
    else
        ref_pos = [interp1(t_traj, traj(:,2), t);
                   interp1(t_traj, traj(:,3), t)];
        ref_vel = [interp1(t_traj, traj(:,4), t);
                   interp1(t_traj, traj(:,5), t)];
    end
end
