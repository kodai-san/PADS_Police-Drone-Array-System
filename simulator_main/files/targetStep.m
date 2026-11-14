function xnext = targetStep(x, P)
% Advance target one time step along the road model.
% Target moves east (+x) until it reaches the intersection (px >= 0),
% then turns south or north depending on P.turnDirection.
%
%   xnext = targetStep(x, P)
%
% Inputs:
%   x - [px; py; vx; vy] target state
%   P - parameter struct with fields: vt, turnDirection, dt
%
% Outputs:
%   xnext - updated target state

px = x(1);

if px < 0
    vx = P.vt;
    vy = 0;
else
    vx = 0;
    if P.turnDirection == "south"
        vy = -P.vt;
    else
        vy =  P.vt;
    end
end

xnext = zeros(4,1);
xnext(1) = x(1) + vx * P.dt;
xnext(2) = x(2) + vy * P.dt;
xnext(3) = vx;
xnext(4) = vy;
end
