function [u_opt, info] = solve_mpc(x0, u_prev, x_ref, mpc, params, walls)
% SOLVE_MPC  Linear MPC with hard wall constraints via quadprog.
%
%   [u_opt, info] = solve_mpc(x0, u_prev, x_ref, mpc, params, walls)
%
%   Constraints enforced:
%     1. Motor bounds     F_min <= F_k <= F_max           (box)
%     2. Pitch limits     |theta_k| <= max_tilt            (linear ineq)
%     3. Wall avoidance   dist(p_k, wall_j) >= margin      (linearised ineq)
%
%   Returns the first control input of the optimal sequence (receding horizon).

    nx = 6;  nu = 2;
    N  = mpc.N;
    dt = mpc.dt_ctrl;

    %% Linearise & discretise
    [A_c, B_c, g_c] = linearise_dynamics(x0, [params.F_hover; params.F_hover], params);
    Ad = eye(nx) + A_c * dt;
    Bd = B_c * dt;
    gd = g_c * dt;

    nz = N * nu;

    %% Power matrices  A^0 … A^N
    A_pow = cell(N+1, 1);
    A_pow{1} = eye(nx);
    for k = 2:N+1
        A_pow{k} = Ad * A_pow{k-1};
    end

    %% Condensed form:  x_stacked = Sx * z + cx
    Sx = zeros(N*nx, nz);
    cx = zeros(N*nx, 1);
    for k = 1:N
        row = (k-1)*nx + (1:nx);
        cx(row) = A_pow{k+1} * x0;
        for j = 0:k-1
            cx(row) = cx(row) + A_pow{k-j} * gd;
        end
        for j = 0:k-1
            col = j*nu + (1:nu);
            Sx(row, col) = A_pow{k-j} * Bd;
        end
    end

    %% Cost matrices
    Q_blk = blkdiag(kron(eye(N-1), mpc.Q), mpc.Qf);
    R_blk = kron(eye(N), mpc.R);

    r = zeros(N*nx, 1);
    for k = 1:N
        r((k-1)*nx + (1:nx)) = x_ref(:, k+1);
    end

    % Input rate cost
    D  = zeros(N*nu, nz);
    d0 = zeros(N*nu, 1);
    for k = 1:N
        row = (k-1)*nu + (1:nu);
        col = (k-1)*nu + (1:nu);
        D(row, col) = eye(nu);
        if k > 1
            D(row, (k-2)*nu + (1:nu)) = -eye(nu);
        else
            d0(row) = -u_prev;
        end
    end
    dR_blk = kron(eye(N), mpc.dR);

    % Assemble QP:  min 0.5 z' H z + f' z
    e = cx - r;
    H = Sx' * Q_blk * Sx + R_blk + D' * dR_blk * D;
    H = (H + H') / 2;
    f = Sx' * Q_blk * e  + D' * dR_blk * d0;

    %% Box constraints (motor bounds)
    lb = repmat(params.F_min, nz, 1);
    ub = repmat(params.F_max, nz, 1);

    %% Pitch constraints:  |theta_k| <= max_tilt
    A_pitch = zeros(2*N, nz);
    b_pitch = zeros(2*N, 1);
    th_idx  = 3;
    for k = 1:N
        rs = (k-1)*nx + th_idx;
        A_pitch(k, :)   =  Sx(rs, :);
        b_pitch(k)      =  params.max_tilt - cx(rs);
        A_pitch(N+k, :) = -Sx(rs, :);
        b_pitch(N+k)    =  params.max_tilt + cx(rs);
    end

    %% Wall constraints (linearised)
    %   dist(p_k, wall_w) >= margin
    %   Linearised about nominal p_k0 = cx_pos:
    %     grad' * Sx_pos * z >= margin - dist0
    %   In standard form (A z <= b):
    %     -grad' * Sx_pos * z <= -(margin - dist0)
    margin  = params.drone_radius + 0.15;
    n_walls = size(walls, 1);

    A_wall_list = zeros(N * n_walls, nz);
    b_wall_list = zeros(N * n_walls, 1);
    nwc = 0;

    for k = 1:N
        rx     = (k-1)*nx + 1;
        pk     = cx(rx:rx+1);
        Sx_pos = Sx(rx:rx+1, :);

        for w = 1:n_walls
            [dist, ~, gd_wall] = point_to_segment_dist_with_grad(pk, walls(w,:));

            if dist < margin * 3   % only activate nearby walls
                nwc = nwc + 1;
                A_wall_list(nwc, :) = -gd_wall' * Sx_pos;
                b_wall_list(nwc)    = -(margin - dist);
            end
        end
    end

    A_wall = A_wall_list(1:nwc, :);
    b_wall = b_wall_list(1:nwc);

    %% Stack all inequality constraints
    A_ineq = [A_pitch; A_wall];
    b_ineq = [b_pitch; b_wall];

    %% Solve
    opts = optimoptions('quadprog', 'Display', 'off', ...
                        'Algorithm', 'interior-point-convex', ...
                        'MaxIterations', 200);

    [z_opt, ~, exitflag] = quadprog(H, f, A_ineq, b_ineq, [], [], lb, ub, [], opts);

    info.exitflag = exitflag;

    if exitflag < 0 || isempty(z_opt)
        warning('MPC infeasible (flag=%d), applying hover.', exitflag);
        u_opt = [params.F_hover; params.F_hover];
        info.feasible = false;
    else
        u_opt = z_opt(1:nu);
        info.feasible = true;
    end
end

%% ========================================================================
%  Private helpers
%  ========================================================================

function [A, B, g] = linearise_dynamics(x0, u0, params)
    m = params.m;  I = params.I;  L = params.L;
    d = params.d_drag;
    th = x0(3);
    Ft = u0(1) + u0(2);

    A = zeros(6);
    A(1,4) = 1;  A(2,5) = 1;  A(3,6) = 1;
    A(4,3) = -Ft/m * cos(th);   A(4,4) = -d;
    A(5,3) = -Ft/m * sin(th);   A(5,5) = -d;

    B = zeros(6,2);
    B(4,1) = -sin(th)/m;  B(4,2) = -sin(th)/m;
    B(5,1) =  cos(th)/m;  B(5,2) =  cos(th)/m;
    B(6,1) = -L/I;         B(6,2) =  L/I;

    f0 = drone_dynamics(x0, u0, params);
    g  = f0 - A*x0 - B*u0;
end

function [dist, closest, grad] = point_to_segment_dist_with_grad(p, seg)
    a  = [seg(1); seg(2)];
    b  = [seg(3); seg(4)];
    ab = b - a;
    ap = p - a;

    len_sq = dot(ab, ab);
    if len_sq < 1e-12
        closest = a;
    else
        t = clamp(dot(ap, ab) / len_sq, 0, 1);
        closest = a + t * ab;
    end

    diff = p - closest;
    dist = norm(diff);
    if dist < 1e-10
        grad = [0; 0];
    else
        grad = diff / dist;
    end
end
