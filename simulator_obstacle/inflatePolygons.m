function inflated = inflatePolygons(polys, r)
% Inflate a cell array of axis-aligned rectangular polygons by radius r.
%
%   inflated = inflatePolygons(polys, r)
%
% Inputs:
%   polys - cell array of Nx2 polygon vertex arrays
%   r     - inflation radius (m)
%
% Outputs:
%   inflated - cell array of inflated polygon vertex arrays

inflated = cell(size(polys));
for i = 1:numel(polys)
    p = polys{i};
    xmin = min(p(:,1)) - r;
    xmax = max(p(:,1)) + r;
    ymin = min(p(:,2)) - r;
    ymax = max(p(:,2)) + r;
    inflated{i} = [xmin, ymin; xmax, ymin; xmax, ymax; xmin, ymax];
end
end
