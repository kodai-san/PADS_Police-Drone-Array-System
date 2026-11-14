function traj = plan_trajectory(t_now, state, tgt_pos, tgt_vel, ...
                                 plan, params, corridor)
% PLAN_TRAJECTORY  Short-horizon corridor-aware path planner.
%
%   traj = plan_trajectory(t_now, state, tgt_pos, tgt_vel, plan, params, corridor)
%
%   Generates a smooth minimum-jerk trajectory from the drone's current
%   state toward the predicted target position, routing through the
%   corridor junction if needed to avoid cutting through walls.
%
%   traj — (N+1)x5 matrix: [t, x, y, vx, vy]

    p0 = state(1:2);
    v0 = state(4:5);

    % Predicted target position at end of horizon
    p_target_pred = tgt_pos + tgt_vel * plan.T_horizon;

    % Determine which branch each party is in
    drone_branch  = get_branch(p0, corridor);
    target_branch = get_branch(p_target_pred, corridor);

    % Junction centre
    jct = [(corridor.vert_x_min + corridor.vert_x_max) / 2, ...
           (corridor.horz_y_min + corridor.horz_y_max) / 2];

    % Build route waypoints
    if drone_branch ~= target_branch
        route_pts = [p0'; jct; p_target_pred'];
    else
        route_pts = [p0'; p_target_pred'];
    end

    % Clamp all points inside corridor
    for i = 1:size(route_pts, 1)
        route_pts(i,:) = clamp_to_corridor(route_pts(i,:)', corridor, params.drone_radius)';
    end

    % Generate smooth trajectory along route
    traj = build_smooth_route(t_now, p0, v0, route_pts, tgt_vel, plan, params, corridor);
end

%% ---- helpers (private to this file) ------------------------------------

function branch = get_branch(p, corridor)
    if p(1) >= corridor.vert_x_min && p(1) <= corridor.vert_x_max ...
            && p(2) < corridor.horz_y_min
        branch = 1;   % vertical corridor
    else
        branch = 2;   % horizontal / junction
    end
end

function p = clamp_to_corridor(p, c, margin)
    if p(2) < c.horz_y_min
        p(1) = clamp(p(1), c.vert_x_min + margin, c.vert_x_max - margin);
        p(2) = clamp(p(2), c.vert_y_min + margin, c.horz_y_min);
    else
        p(1) = clamp(p(1), c.vert_x_min + margin, c.horz_x_max - margin);
        p(2) = clamp(p(2), c.horz_y_min + margin, c.horz_y_max - margin);
    end
end

function traj = build_smooth_route(t_now, p0, v0, route_pts, tgt_vel, plan, params, corridor)

    p_goal = route_pts(end,:)';
    v_goal = tgt_vel;
    T      = plan.T_horizon;
    traj   = zeros(plan.N_steps + 1, 5);

    if size(route_pts, 1) <= 2
        % Single segment
        for ax = 1:2
            [pc, vc] = min_jerk_coeffs(p0(ax), v0(ax), 0, ...
                                        p_goal(ax), v_goal(ax), 0, T);
            for k = 0:plan.N_steps
                tau = k * plan.dt_plan;
                traj(k+1, 1)    = t_now + tau;
                traj(k+1, 1+ax) = polyval(pc, tau);
                traj(k+1, 3+ax) = polyval(vc, tau);
            end
        end
    else
        % Multi-segment: split time proportional to distance
        n_seg   = size(route_pts, 1) - 1;
        seg_len = zeros(n_seg, 1);
        for i = 1:n_seg
            seg_len(i) = norm(route_pts(i+1,:) - route_pts(i,:));
        end
        seg_time = seg_len / sum(seg_len) * T;
        seg_time = max(seg_time, 0.3);
        seg_time = seg_time / sum(seg_time) * T;

        t_seg_start = 0;
        v_start     = v0;

        for s = 1:n_seg
            p_start = route_pts(s,:)';
            p_end   = route_pts(s+1,:)';
            T_seg   = seg_time(s);

            if s == n_seg
                v_end = v_goal;
            else
                dir_next = route_pts(s+2,:)' - route_pts(s+1,:)';
                if norm(dir_next) > 0.01
                    v_end = dir_next / norm(dir_next) * norm(v_goal);
                else
                    v_end = [0; 0];
                end
            end

            for ax = 1:2
                [pc, vc] = min_jerk_coeffs(p_start(ax), v_start(ax), 0, ...
                                            p_end(ax), v_end(ax), 0, T_seg);
                for k = 0:plan.N_steps
                    tau_abs   = k * plan.dt_plan;
                    tau_local = tau_abs - t_seg_start;
                    if tau_local >= 0 && tau_local <= T_seg + 1e-6
                        tau_local = min(tau_local, T_seg);
                        traj(k+1, 1)    = t_now + tau_abs;
                        traj(k+1, 1+ax) = polyval(pc, tau_local);
                        traj(k+1, 3+ax) = polyval(vc, tau_local);
                    end
                end
            end

            t_seg_start = t_seg_start + T_seg;
            v_start     = v_end;
        end

        % Fill any remaining time entries
        for k = 0:plan.N_steps
            if traj(k+1, 1) == 0
                traj(k+1, 1) = t_now + k * plan.dt_plan;
            end
        end
    end

    % Clamp all trajectory points inside corridor
    for k = 1:size(traj, 1)
        p_k = clamp_to_corridor(traj(k, 2:3)', corridor, params.drone_radius);
        traj(k, 2:3) = p_k';
    end
end
