function dstate = drone_dynamics(state, u, params)
% DRONE_DYNAMICS  2D quadcopter equations of motion.
%
%   dstate = drone_dynamics(state, u, params)
%
%   State:  [x, y, theta, vx, vy, omega]
%   Input:  [F1, F2]  (left / right motor thrust)
%
%   Convention:
%     - Thrust acts along the body y-axis: world direction [-sin(th), cos(th)]
%     - Positive theta = counter-clockwise pitch
%     - Torque = (F2 - F1) * L

    th = state(3);
    vx = state(4);
    vy = state(5);

    Ft    = u(1) + u(2);
    tau   = (u(2) - u(1)) * params.L;

    ax    = -Ft / params.m * sin(th) - params.d_drag * vx;
    ay    =  Ft / params.m * cos(th) - params.g - params.d_drag * vy;
    alpha =  tau / params.I;

    dstate = [vx; vy; state(6); ax; ay; alpha];
end
