function [Phi, Gamma] = buildPredictionMatrices(A, B, N)
% Build stacked prediction matrices for the MPC horizon.
%
%   [Phi, Gamma] = buildPredictionMatrices(A, B, N)
%
%   x_stacked = Phi * x0 + Gamma * U_stacked
%
% where x_stacked is [(N+1)*nx x 1] and U_stacked is [N*nu x 1].
%
% Inputs:
%   A  - [nx x nx] state transition matrix
%   B  - [nx x nu] input matrix
%   N  - prediction horizon length
%
% Outputs:
%   Phi   - [(N+1)*nx  x  nx]   state propagation matrix
%   Gamma - [(N+1)*nx  x  N*nu] input influence matrix

nx = size(A, 1);
nu = size(B, 2);

Phi   = zeros(nx*(N+1), nx);
Gamma = zeros(nx*(N+1), nu*N);

Phi(1:nx, :) = eye(nx);
for i = 1:N
    rows = i*nx + (1:nx);
    Phi(rows, :) = A^i;
    for j = 1:i
        cols = (j-1)*nu + (1:nu);
        Gamma(rows, cols) = A^(i-j) * B;
    end
end
end
