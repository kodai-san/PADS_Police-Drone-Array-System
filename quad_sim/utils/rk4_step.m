function state_new = rk4_step(state, u, dt, params)
% RK4_STEP  4th-order Runge-Kutta integration of drone dynamics.
%
%   state_new = rk4_step(state, u, dt, params)

    k1 = drone_dynamics(state,            u, params);
    k2 = drone_dynamics(state + dt/2 * k1, u, params);
    k3 = drone_dynamics(state + dt/2 * k2, u, params);
    k4 = drone_dynamics(state + dt   * k3, u, params);

    state_new = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4);
end
