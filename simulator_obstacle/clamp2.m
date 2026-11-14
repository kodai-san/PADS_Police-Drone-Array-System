function v = clamp2(v, vmax)
% Clamp the magnitude of a 2D vector to vmax.
%
%   v = clamp2(v, vmax)

n = norm(v);
if n > vmax
    v = v * (vmax / (n + 1e-12));
end
end
