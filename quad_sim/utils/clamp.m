function y = clamp(x, lo, hi)
% CLAMP  Constrain value to [lo, hi].
    y = max(lo, min(hi, x));
end
