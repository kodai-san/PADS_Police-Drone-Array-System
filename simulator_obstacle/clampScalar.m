function s = clampScalar(s, smaxAbs)
% Clamp scalar s to the range [-smaxAbs, +smaxAbs].
%
%   s = clampScalar(s, smaxAbs)

s = max(-smaxAbs, min(smaxAbs, s));
end
