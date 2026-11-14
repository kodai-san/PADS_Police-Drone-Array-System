function tf = segmentsIntersect(p1, p2, q1, q2)
% Test whether two line segments (p1-p2) and (q1-q2) intersect.
%
%   tf = segmentsIntersect(p1, p2, q1, q2)
%
% Each input is a [1x2] point [x y].

tf = false;

d1 = crossDirection(q1, q2, p1);
d2 = crossDirection(q1, q2, p2);
d3 = crossDirection(p1, p2, q1);
d4 = crossDirection(p1, p2, q2);

% Standard proper crossing
if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ...
    ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)))
    tf = true;
    return;
end

% Collinear / endpoint-on-segment cases
if d1 == 0 && onSegment(q1, q2, p1), tf = true; return; end
if d2 == 0 && onSegment(q1, q2, p2), tf = true; return; end
if d3 == 0 && onSegment(p1, p2, q1), tf = true; return; end
if d4 == 0 && onSegment(p1, p2, q2), tf = true; return; end
end

function d = crossDirection(a, b, c)
% Signed cross product direction of (c - a) relative to (b - a).
val = (c(1)-a(1))*(b(2)-a(2)) - (c(2)-a(2))*(b(1)-a(1));
if abs(val) < 1e-12
    d = 0;
else
    d = sign(val);
end
end

function tf = onSegment(a, b, c)
% Check if point c lies on segment a-b (assuming collinear).
tf = (min(a(1),b(1)) - 1e-12 <= c(1) && c(1) <= max(a(1),b(1)) + 1e-12 && ...
      min(a(2),b(2)) - 1e-12 <= c(2) && c(2) <= max(a(2),b(2)) + 1e-12);
end
