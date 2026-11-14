function [corridor, walls] = build_corridor()
% BUILD_CORRIDOR  Define the T-shaped corridor geometry and wall segments.
%
%   [corridor, walls] = build_corridor()
%
%   corridor — struct with corridor dimensions
%   walls    — Nx4 matrix, each row [x1,y1,x2,y2] is a wall segment
%
%   Layout (top-down):
%
%       horz_y_max ______________________________________
%                  |                                     |
%                  |       HORIZONTAL CORRIDOR            |
%                  |_____________________________________|
%       horz_y_min        |               |
%                         |               |
%                  vert_x_min      vert_x_max
%                         |               |
%                         | VERT CORRIDOR  |
%                         |               |
%                         |_______________|
%                        vert_y_min

    %% Dimensions
    corridor.vert_x_min = 8.0;
    corridor.vert_x_max = 12.0;
    corridor.vert_y_min = 0.0;
    corridor.vert_y_max = 23.0;   % extends up to join horizontal

    corridor.horz_x_min = 8.0;    % left edge (aligns with vert_x_min)
    corridor.horz_x_max = 45.0;
    corridor.horz_y_min = 17.0;
    corridor.horz_y_max = 23.0;

    corridor.junction_y = 20.0;   % centre of junction

    %% Wall segments [x1, y1, x2, y2]
    walls = [
        % Bottom of vertical corridor
        corridor.vert_x_min, corridor.vert_y_min, corridor.vert_x_max, corridor.vert_y_min;

        % Left wall of vertical corridor (bottom → junction)
        corridor.vert_x_min, corridor.vert_y_min, corridor.vert_x_min, corridor.horz_y_min;

        % Right wall of vertical corridor (bottom → junction)
        corridor.vert_x_max, corridor.vert_y_min, corridor.vert_x_max, corridor.horz_y_min;

        % Bottom of horizontal corridor (right of vertical, junction corner → right end)
        corridor.vert_x_max, corridor.horz_y_min, corridor.horz_x_max, corridor.horz_y_min;

        % Right end of horizontal corridor
        corridor.horz_x_max, corridor.horz_y_min, corridor.horz_x_max, corridor.horz_y_max;

        % Top of horizontal corridor (right end → left end)
        corridor.horz_x_max, corridor.horz_y_max, corridor.vert_x_min, corridor.horz_y_max;

        % Left wall of horizontal corridor (top → bottom, closes the T)
        corridor.vert_x_min, corridor.horz_y_max, corridor.vert_x_min, corridor.horz_y_min;
    ];
end
