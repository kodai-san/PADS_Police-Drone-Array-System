function Du = diffMatrix2D(N)
% Build first-difference matrix for 2D input sequence (for input smoothing).
%
%   Du = diffMatrix2D(N)
%
% Du * U gives [u2-u1; u3-u2; ... ; uN-u(N-1)] where each u is 2D.
%
% Inputs:
%   N  - number of control steps
%
% Outputs:
%   Du - [2*(N-1) x 2*N] difference matrix

Du = zeros(2*(N-1), 2*N);
for k = 1:N-1
    rows = 2*k-1 : 2*k;
    colsPrev = 2*(k-1)+1 : 2*(k-1)+2;
    colsCurr = 2*k+1     : 2*k+2;
    Du(rows, colsPrev) = -eye(2);
    Du(rows, colsCurr) =  eye(2);
end
end
