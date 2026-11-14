function tf = pointInAnyBuilding(p, buildingPolys)
% Check if point p is inside any building polygon.
tf = false;
for i = 1:numel(buildingPolys)
    if pointInPoly(p, buildingPolys{i})
        tf = true;
        return;
    end
end
end
