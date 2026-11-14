function U = solveMPC_QP(xd, u_prev, pref, vref, Phi, Gamma, Du, P, W)
% Solve the MPC quadratic program for planar acceleration commands.
%
%   U = solveMPC_QP(xd, u_prev, pref, vref, Phi, Gamma, Du, P, W)
%
% Minimises:
%   sum_k || x_k - xdes_k ||^2_Q  +  || U ||^2_R  +  w_du * || Delta U ||^2
%
% Subject to:
%   -amax <= u_k <= amax  (component-wise)
%
% Inputs:
%   xd     - [4x1] current drone state [px; py; vx; vy]
%   u_prev - [2x1] previous acceleration command
%   pref   - [2 x (N+1)] reference positions over horizon
%   vref   - [2 x (N+1)] reference velocities over horizon
%   Phi    - prediction state matrix
%   Gamma  - prediction input matrix
%   Du     - first-difference matrix for input sequence
%   P      - parameter struct (fields: N, amax)
%   W      - weight struct (fields: w_pos, w_vel, w_u, w_du)
%
% Outputs:
%   U - [2*N x 1] optimal acceleration sequence, or NaN if infeasible

nx = 4;
nu = 2;
N  = P.N;

% Build stacked desired state vector
xdes = zeros(nx*(N+1), 1);
for i = 1:N+1
    idx = (i-1)*nx + (1:nx);
    xdes(idx) = [pref(:,i); vref(:,i)];
end

% Stage cost weight matrix (position + velocity tracking)
Qk = diag([W.w_pos, W.w_pos, W.w_vel, W.w_vel]);
Q  = kron(eye(N+1), Qk);

% Input cost
R = W.w_u * eye(nu*N);

% Input smoothing cost
wdu = W.w_du;
H = (Gamma' * Q * Gamma) + R + wdu * (Du' * Du);

% First-step smoothing: penalise (u_1 - u_prev)
E0 = zeros(nu, nu*N);
E0(:, 1:nu) = eye(nu);
H = H + wdu * (E0' * E0);

% Linear term
f = Gamma' * Q * (Phi * xd - xdes) - wdu * (E0' * u_prev);

% Bounds (component-wise acceleration limits)
lb = -P.amax * ones(nu*N, 1);
ub =  P.amax * ones(nu*N, 1);

% Solve
opts = optimoptions('quadprog', 'Display', 'off');
[U, ~, exitflag] = quadprog((H + H')/2, f, [], [], [], [], lb, ub, [], opts);

if exitflag <= 0
    U = nan(nu*N, 1);
end
end
