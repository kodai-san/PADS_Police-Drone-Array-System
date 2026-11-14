function d = min_wall_distance(p, walls)
% MIN_WALL_DISTANCE  Minimum distance from point p to any wall segment.
%
%   d = min_wall_distance(p, walls)
%
%   p     — 2x1 position
%   walls — Nx4 matrix [x1,y1,x2,y2]

    d = inf;
    for i = 1:size(walls, 1)
        a  = walls(i, 1:2)';
        b  = walls(i, 3:4)';
        ab = b - a;
        ap = p - a;
        len_sq = dot(ab, ab);
        if len_sq < 1e-12
            di = norm(p - a);
        else
            t  = max(0, min(1, dot(ap, ab) / len_sq));
            di = norm(p - (a + t * ab));
        end
        d = min(d, di);
    end
end
