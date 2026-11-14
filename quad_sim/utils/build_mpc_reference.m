function x_ref = build_mpc_reference(t_now, planned_traj, mpc, params)
% BUILD_MPC_REFERENCE  Construct (6 x N+1) state reference for MPC.
%
%   x_ref = build_mpc_reference(t_now, planned_traj, mpc, params)
%
%   Samples planned trajectory at MPC timesteps and computes feedforward
%   pitch angle from the reference acceleration profile.

    nx = 6;
    N  = mpc.N;
    dt = mpc.dt_ctrl;

    x_ref = zeros(nx, N+1);

    for k = 0:N
        tk = t_now + k * dt;
        [p_ref, v_ref] = sample_trajectory(tk, planned_traj);

        % Feedforward pitch from finite-difference acceleration
        dt_fd = 0.05;
        [~, v_next] = sample_trajectory(tk + dt_fd, planned_traj);
        a_ref  = (v_next - v_ref) / dt_fd;
        th_ref = atan2(-a_ref(1), a_ref(2) + params.g);
        th_ref = clamp(th_ref, -params.max_tilt, params.max_tilt);

        x_ref(:, k+1) = [p_ref(1); p_ref(2); th_ref; v_ref(1); v_ref(2); 0];
    end
end
