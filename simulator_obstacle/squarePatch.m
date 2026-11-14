function [sx, sy] = squarePatch(cx, cy, side)
% Return vertices of an axis-aligned square centred at (cx, cy).
%
%   [sx, sy] = squarePatch(cx, cy, side)
%
% Outputs:
%   sx, sy - [1x4] vertex coordinate arrays for use with patch()

h = side / 2;
sx = [cx-h, cx+h, cx+h, cx-h];
sy = [cy-h, cy-h, cy+h, cy+h];
end
