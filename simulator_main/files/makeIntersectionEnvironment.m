function env = makeIntersectionEnvironment(roadW)
% Create a simple intersection environment with four building blocks.
%
%   env = makeIntersectionEnvironment(roadW)
%
% Inputs:
%   roadW - total road width (m)
%
% Outputs:
%   env   - struct with fields: buildings, roadHalfW, bounds

w = roadW / 2;
Xmin = -60; Xmax = 60;
Ymin = -60; Ymax = 60;

% Four quadrant buildings (counter-clockwise vertex order)
b1 = [Xmin,  w;  -w,  w;  -w, Ymax;  Xmin, Ymax];  % upper-left
b2 = [  w,   w; Xmax,  w; Xmax, Ymax;    w, Ymax];  % upper-right
b3 = [Xmin, Ymin;  -w, Ymin;  -w, -w;  Xmin, -w];   % lower-left
b4 = [  w,  Ymin; Xmax, Ymin; Xmax, -w;    w, -w];   % lower-right

env.buildings = {b1, b2, b3, b4};
env.roadHalfW = w;
env.bounds    = [Xmin, Xmax, Ymin, Ymax];
end
