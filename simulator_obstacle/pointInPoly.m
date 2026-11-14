function tf = pointInPoly(p, poly)
% Ray-casting point-in-polygon test.
%
%   tf = pointInPoly(p, poly)
%
% Inputs:
%   p    - [2x1] or [1x2] point
%   poly - [Nx2] polygon vertices

x = p(1);
y = p(2);
n = size(poly, 1);
inside = false;
j = n;

for i = 1:n
    xi = poly(i,1); yi = poly(i,2);
    xj = poly(j,1); yj = poly(j,2);

    crosses = ((yi > y) ~= (yj > y)) && ...
              (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi);
    if crosses
        inside = ~inside;
    end
    j = i;
end

tf = inside;
end
