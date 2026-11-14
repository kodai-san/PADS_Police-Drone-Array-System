function los = hasLineOfSight(p1, p2, buildingPolys)
% Check if there is a clear line of sight between p1 and p2
% (no building polygon blocks the segment).
%
%   los = hasLineOfSight(p1, p2, buildingPolys)

los = true;
seg = [p1(:)'; p2(:)'];
midpoint = (p1(:) + p2(:))' / 2;

for i = 1:numel(buildingPolys)
    poly = buildingPolys{i};
    if segmentIntersectsPolygon(seg, poly) || pointInPoly(midpoint', poly)
        los = false;
        return;
    end
end
end
