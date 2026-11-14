function tf = segmentIntersectsPolygon(seg, poly)
% Check if a line segment intersects any edge of a polygon.
%
%   tf = segmentIntersectsPolygon(seg, poly)
%
% Inputs:
%   seg  - [2x2] rows are endpoints [x y]
%   poly - [Nx2] polygon vertices

p = seg(1,:);
q = seg(2,:);
n = size(poly, 1);
tf = false;

for k = 1:n
    a = poly(k, :);
    b = poly(mod(k, n) + 1, :);
    if segmentsIntersect(p, q, a, b)
        tf = true;
        return;
    end
end
end
