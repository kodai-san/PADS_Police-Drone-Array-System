function [phi, theta] = tiltStep(phi, theta, phi_cmd, theta_cmd, P)
% First-order tilt response with rate clamp
%
%   [phi, theta] = tiltStep(phi, theta, phi_cmd, theta_cmd, P)
%
% Inputs:
%   phi, theta      - current roll/pitch angles (rad)
%   phi_cmd, theta_cmd - commanded roll/pitch angles (rad)
%   P               - parameter struct with fields: tiltTau, tiltRateMax, dt, tiltMax
%
% Outputs:
%   phi, theta      - updated roll/pitch angles (rad)

phi_dot   = (phi_cmd - phi) / P.tiltTau;
theta_dot = (theta_cmd - theta) / P.tiltTau;

phi_dot   = clampScalar(phi_dot,   P.tiltRateMax);
theta_dot = clampScalar(theta_dot, P.tiltRateMax);

phi   = phi   + P.dt * phi_dot;
theta = theta + P.dt * theta_dot;

phi   = clampScalar(phi,   P.tiltMax);
theta = clampScalar(theta, P.tiltMax);
end
