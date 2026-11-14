function [pos_c, vel_c] = min_jerk_coeffs(p0, v0, a0, pf, vf, af, T)
% MIN_JERK_COEFFS  5th-order polynomial coefficients for minimum-jerk trajectory.
%
%   [pos_c, vel_c] = min_jerk_coeffs(p0, v0, a0, pf, vf, af, T)
%
%   Boundary conditions: position, velocity, acceleration at t=0 and t=T.
%   Returns coefficient vectors in MATLAB polyval order (highest power first).

    c0 = p0;  c1 = v0;  c2 = a0 / 2;
    T2 = T^2;  T3 = T^3;  T4 = T^4;  T5 = T^5;

    A = [  T3,    T4,    T5;
          3*T2,  4*T3,  5*T4;
          6*T,  12*T2, 20*T3];

    b = [pf - c0 - c1*T - c2*T2;
         vf - c1 - 2*c2*T;
         af - 2*c2];

    x = A \ b;

    pos_c = [x(3), x(2), x(1), c2, c1, c0];
    vel_c = [5*x(3), 4*x(2), 3*x(1), 2*c2, c1];
end
