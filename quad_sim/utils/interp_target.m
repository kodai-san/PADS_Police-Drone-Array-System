function p = interp_target(t, t_vec, pos_vec)
% INTERP_TARGET  Query precomputed target position at time t.
%
%   p = interp_target(t, t_vec, pos_vec)

    t = max(t_vec(1), min(t_vec(end), t));
    p = [interp1(t_vec, pos_vec(1,:), t);
         interp1(t_vec, pos_vec(2,:), t)];
end
