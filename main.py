import numpy as np  # Only used for graphing
import matplotlib.pyplot as plt  # Only used for graphing
import matplotlib.animation as animation
import math
import keyboard


files = ["kvadr.scad", "6stran_podstava.scad", "Big_flat_house.scad", "motylek.scad", "stair_house.scad",
           "C_house.scad", "U_house_flat.scad", "weird_house.scad"]
file_number = 5 # 0 to 7

limits = [(0, 10, 0, 7), (0, 12, 0, 15), (0, 20, 0, 35), (-9, 12, -3, 15), (0, 12, 0, 15), (0, 20, -5, 10),
          (0, 12, 0, 15), (0, 16, 0, 13)]
#xmin, xmax, ymin, ymax

height = 0
f = open(files[file_number], "r")
f = f.read()
drawing = True
printing = True
unpause = False
roof_points = []
used_lines = []
animation_points = [[], []]
animation_lines = [[], []]
animation_axes = [[], []]
border_points = []
animation_draw_yellow_axes = []
number_of_frames = -1
rounding_error = 0.01
angle = math.radians(45)


class Line:
    def __init__(self, point1: list, point2: list, beginning: list, border: bool, final: bool, walls: list[int]):
        self.point1 = point1
        self.point2 = point2
        self.beginning = beginning
        self.border = border
        self.final = final
        self.walls = walls
        self.enclose_points = []
        self.banned = False

    def slope(self):
        if self.point1[0] < self.point2[0]:
            try:
                slope = (self.point2[1] - self.point1[1]) / (self.point2[0] - self.point1[0])
                return slope
            except ZeroDivisionError:
                return 999999999999999
        else:
            try:
                slope = (self.point1[1] - self.point2[1]) / (self.point1[0] - self.point2[0])
                return slope
            except ZeroDivisionError:
                return 999999999999999

    def __eq__(self, other):
        if self.point1 == other.point1 and self.point2 == other.point2 and self.beginning == \
                other.beginning and self.border == other.border:
            return True
        return False

    def __str__(self):
        return str(self.point1) + str(self.point2) + str(self.beginning) + str(self.border) + str(self.final) \
               + str(self.walls) + str(self.banned)

    def lenght(self):
        return math.sqrt(math.pow((self.point1[0] - self.point2[0]), 2) + math.pow(self.point1[1] - self.point2[1], 2))


fig, ax = plt.subplots(figsize=(20, 20))


def start_drawing():
    global limits, file_number, fig, ax
    xmin, xmax, ymin, ymax = limits[file_number]
    ticks_frequency = 5

    ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect='equal')

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

    # Create minor ticks placed at each integer to enable drawing of minor grid
    # lines: note that this has no effect in this example with ticks_frequency=1
    ax.set_xticks(np.arange(xmin, xmax + 1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax + 1), minor=True)

    # Draw major and minor grid lines
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot(1, 0, marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot(0, 1, marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)


def change_drawing(ax):
    global limits, file_number

    xmin, xmax, ymin, ymax = limits[file_number]
    ticks_frequency = 5

    ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect='equal')

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

    # Create minor ticks placed at each integer to enable drawing of minor grid
    # lines: note that this has no effect in this example with ticks_frequency=1
    ax.set_xticks(np.arange(xmin, xmax + 1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax + 1), minor=True)

    # Draw major and minor grid lines
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot(1, 0, marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot(0, 1, marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)


def draw(points: list):
    global limits, file_number
    xs = []
    ys = []
    colors = ['m']
    for toppp_point in points:
        xs.append(toppp_point[0])
        ys.append(toppp_point[1])
    """
    for line_point in points:
        if line_point == points[0]:
            plt.plot([line_point[0], points[points.index(line_point) + 1][0]], [line_point[1],
                                                                               points[points.index(line_point) + 1][1]],
                    c='m')
        elif line_point == points[len(points) - 1]:
            plt.plot([line_point[0], points[0][0]], [line_point[1], points[0][1]], c='m')
        else:
            plt.plot([line_point[0], points[points.index(line_point) + 1][0]], [line_point[1],
                                                                               points[points.index(line_point) + 1][1]],
                    c='m')"""


def find_points(no_list: str) -> list:
    points = []
    no_list = no_list[:no_list.index(";") - 1]
    no_list = no_list[no_list.index("[") + 1:]
    no_list = no_list + ",L"
    coor = []
    for i in range(no_list.count("[")):
        for j in range(3):
            temp = no_list[:no_list.index(",")]
            no_list = no_list[no_list.index(",") + 1:]
            if j == 0:
                temp = temp[temp.find("[") + 1:]
            if j == 2:
                temp = temp[:-1]
            temp = float(temp)
            coor.append(temp)
            if j == 2:
                points.append(coor)
                coor = []

    return points


"""
Both of these functions are just to find the points and the faces respectively, .scad is a pretty rare format to work 

with in python so I didn't find a library to do this.
"""


def find_faces(no_list: str) -> list:
    faces = []
    no_list = no_list[no_list.index("Faces"):]
    no_list = no_list[no_list.index("[") + 1:no_list.index(";") - 1]
    no_list = no_list + ",L"
    perin_count = no_list.count("[")
    int(no_list.count(","))
    for i in range(perin_count):
        coor = []
        temp = no_list[no_list.index("[") + 1:no_list.index("],")]
        temp = temp + ","
        number = temp.count(",")
        for j in range(number):
            num = int(temp[:temp.find(',')])
            temp = temp[temp.find(',') + 1:]
            coor.append(num)
        faces.append(coor)
        if i != perin_count - 1:
            no_list = no_list[no_list.index("],"):]
            no_list = no_list[no_list.index("["):]
    return faces


def find_height(no_list: str):
    global height
    if no_list.count("roof_height") != 1:
        return None
    no_list = no_list[no_list.index("roof_height"):]
    no_list = no_list[no_list.index("=") + 1:]
    no_list = no_list[:no_list.index(";")]
    no_list = float(no_list)
    return no_list


def find_roof_angle(no_list: str):
    global angle
    if no_list.count("roof_angle") != 1:
        return None
    no_list = no_list[no_list.index("roof_angle"):]
    no_list = no_list[no_list.index("=") + 1:]
    no_list = no_list[:no_list.index(";")]
    no_list = float(no_list)
    no_list = math.radians(no_list)
    return no_list


def find_banned_walls(no_list: str):
    global Lines
    no_list = no_list[no_list.index("banned_walls"):]
    no_list = no_list[no_list.index("=") + 1:]
    no_list = no_list[:no_list.index(";")]
    found_walls = []
    if no_list.find(',') < 1:
        no_list = no_list[no_list.find('[') + 1:no_list.find(']')]
        no_list = int(no_list)
        found_walls.append(no_list)
    else:
        no_list = no_list[no_list.find('[') + 1:]
        changed_list = no_list
        for i in range(no_list.count(',') + 1):
            if i == no_list.count(','):
                changed_list = changed_list[:changed_list.find(']')]
                found_walls.append(int(changed_list))
            else:
                found_walls.append(int(changed_list[:changed_list.find(',')]))
                changed_list = changed_list[changed_list.find(',') + 1:]
    for banned_wall in found_walls:
        Lines[banned_wall].banned = True


def not_top_faces(points: list, faces: list) -> list:
    # Gets rid of the top face, as it will get covered by the roof
    global height, f
    heights = []
    bad_points = []
    good_faces = []
    for height_point in points:
        if height_point[2] not in heights:
            heights.append(height_point[2])
    top = max(heights)
    for i in range(len(points)):
        if not (points[i][2] >= top):
            bad_points.append(points[i])
    for face in faces:
        for height_point in face:
            if points[height_point] in bad_points:
                good_faces.append(face)
                break

    if find_height(f):
        height = heights[len(heights) - 1] + (heights[len(heights) - 1] - heights[0]) * find_height(f)
    else:
        height = heights[len(heights) - 1] + heights[len(heights) - 1] - heights[0]
    return good_faces


def worthless_points(points: list) -> list:
    # Dump points that don't appear in the top layer
    heights = []
    bad_points = []
    for height_point in points:
        if height_point[2] not in heights:
            heights.append(height_point[2])
    top = max(heights)
    for i in range(len(points)):
        if not (points[i][2] >= top):
            bad_points.append(points[i])
    for bad_point in bad_points:
        points.remove(bad_point)
    return bad_points


def add_top_points(points: list[list], tallness: int or float) -> list:
    upper_points = []
    for bottom_point in points:
        new_point = [bottom_point[0], bottom_point[1], bottom_point[2] + tallness]
        upper_points.append(new_point)
    return upper_points


def create_faces(low_points: list[list]):
    global Faces
    face = []
    if len(Faces) != 1:
        for low_point in low_points:
            face.append(low_points.index(low_point))
        Faces = [face]
    for second_point in low_points:
        if low_points.index(second_point) == len(low_points) - 1:
            face = [low_points.index(second_point), 2 * len(low_points) - 1, len(low_points), 0]
        else:
            face = [low_points.index(second_point), len(low_points) + low_points.index(second_point), len(low_points) +
                    low_points.index(second_point) + 1, low_points.index(second_point) + 1]
        Faces.append(face)


def find_angle(a: list, mid: list, c: list):
    """Finds the angle axis by converting the two lines making up the angle into vectors, making them the same length

    and adding them up. The resulting line goes the same distance both ways from its beginning.
    """
    global height
    xa = a[0] - mid[0]
    ya = a[1] - mid[1]
    xmid = mid[0]
    ymid = mid[1]
    zmid = mid[2]
    xc = c[0] - mid[0]
    yc = c[1] - mid[1]
    len_a = math.sqrt(math.pow(xa, 2) + math.pow(ya, 2))
    len_c = math.sqrt(math.pow(xc, 2) + math.pow(yc, 2))
    ratio = len_a / len_c
    if ratio != 1:
        xc = xc * ratio
        yc = yc * ratio
    return Line([xc + xa + xmid, ya + yc + ymid, height], [- xc - xa + xmid, - ya - yc + ymid, height],
                [xmid, ymid, zmid], False, False, [0, 0])


def border_lines(points: list) -> list:
    global animation_lines
    lines = []
    for border_point in points:
        walls = [Points.index(border_point) + 1, Points.index(border_point) + 1]
        if points.index(border_point) == len(points) - 1:
            lines.append(Line(border_point, Points[0], border_point, True, False, walls))
            animation_lines[0].append(Line(border_point, Points[0], border_point, True, False, walls))
            animation_lines[1].append(Line(border_point, Points[0], border_point, True, False, walls))
        else:
            lines.append(Line(border_point, Points[Points.index(border_point) + 1], border_point, True, False, walls))
            animation_lines[0].append(Line(border_point, Points[Points.index(border_point) + 1], border_point, True, False, walls))
            animation_lines[1].append(
                Line(border_point, Points[Points.index(border_point) + 1], border_point, True, False, walls))
    return lines


def setup_angles(points: list):
    global Lines, animation_draw_yellow_axes, animation_points, animation_lines
    # Just a setup for the find_angle function.
    for border_point in points:
        wall_lines = []
        cont = False
        for checked_line in Lines:
            if checked_line.point1 == border_point or checked_line.point2 == border_point:
                if checked_line.banned:
                    cont = True
        if cont:
            continue
        if points.index(border_point) == 0:
            angle_line = find_angle(points[len(points) - 1], points[0], points[1])
        elif points.index(border_point) == len(points) - 1:
            angle_line = find_angle(points[len(points) - 2], points[len(points) - 1], points[0])
        else:
            angle_line = find_angle(points[points.index(border_point) - 1], points[points.index(border_point)],
                                    points[points.index(border_point) + 1])
        for wall_line in Lines:
            if wall_line.border and (wall_line.point1 == border_point or wall_line.point2 == border_point):
                wall_lines.append(wall_line)
        if len(wall_lines) == 2:
            walls = get_walls(wall_lines[0], wall_lines[1])
            angle_line.walls = walls
        animation_draw_yellow_axes.append(angle_line)
        Lines.append(angle_line)
        animation_lines[1].append(Line(angle_line.point1, angle_line.point2, angle_line.beginning, angle_line.border, angle_line.final, angle_line.walls))
        animation_points[0].append(angle_line.beginning)
        animation_points[1].append(angle_line.beginning)



def check_crossing(point1: list[int], point2: list[int]) -> bool:
    global Lines, rounding_error
    temp_line = Line(point1, point2, point1, False, False, [0, 0])
    border_lins = []
    if not check_in_shape([(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2, point1[2]]):
        return True
    for border_line in Lines:
        if border_line.point1 == point1 or border_line.point2 == point1 or border_line.point1 == point2 or \
                border_line.point2 == point2:
            continue
        if border_line.border:
            border_lins.append(border_line)
    for border_line in border_lins:
        if type(line_intersecting(temp_line, border_line)) == bool:
            continue
        intersect = line_intersecting(temp_line, border_line)
        if abs(distance(border_line.point1, border_line.point2) - (distance(border_line.point1, intersect) +
                                                                   distance(border_line.point2,
                                                                            intersect))) > 2 * rounding_error:
            continue
        if abs(distance(point1, point2) - (distance(point1, intersect) + distance(point2, intersect))) < \
                2 * rounding_error:
            return True
    return False


def find_lines():
    global Points, Lines, roof_points
    axis_lines = []
    for axis_line in Lines:
        if not axis_line.border:
            axis_lines.append(axis_line)
    find_intersects(axis_lines)


def find_intersects(axis_lines: list[Line]):
    global Points, Lines, rounding_error, roof_points
    intersected_lines = []
    for axis_line in axis_lines:
        intersects = [axis_line]
        for second_line in axis_lines:
            if axis_line == second_line:
                continue
            if axis_line.final:
                continue
            if type(line_intersecting(second_line, axis_line)) == bool and line_intersecting(second_line, axis_line):
                intersects.append(second_line)
                continue
            if type(line_intersecting(second_line, axis_line)) != bool:
                if check_crossing(axis_line.beginning, line_intersecting(second_line, axis_line)):
                    continue
                if check_crossing(second_line.beginning, line_intersecting(second_line, axis_line)):
                    continue
                if not check_in_shape(line_intersecting(second_line, axis_line)):
                    continue
                intersects.append(second_line)
        intersected_lines.append(intersects)
    for axis_list in intersected_lines:
        if len(axis_list) == 1:
            continue
        axis_line = axis_list[0]
        lenght = 0
        for second_line in axis_list:
            if type(line_intersecting(second_line, axis_line)) != bool:
                lenght = distance(axis_line.beginning, line_intersecting(second_line, axis_line))
                break
        if lenght == 0:
            continue
        for second_line in axis_list:
            if type(line_intersecting(second_line, axis_line)) != bool:
                if distance(axis_line.beginning, line_intersecting(second_line, axis_line)) <= lenght + rounding_error:
                    lenght = distance(axis_line.beginning, line_intersecting(second_line, axis_line))
        remove_lines = []

        for second_line in axis_list:
            if type(line_intersecting(second_line, axis_line)) != bool:
                if distance(axis_line.beginning, line_intersecting(second_line, axis_line)) >= lenght + rounding_error:
                    remove_lines.append(second_line)
        for second_line in remove_lines:
            axis_list.remove(second_line)

    for check_list in intersected_lines:
        if len(check_list) == 1:
            continue
        axis_line = check_list[0]
        remove_list = []
        for second_line in check_list:
            second_list = []
            if second_line == axis_line:
                continue
            for a_list in intersected_lines:
                if a_list[0] == second_line:
                    second_list = a_list
            if axis_line not in second_list:
                remove_list.append(second_line)
        for remove_line in remove_list:
            check_list.remove(remove_line)

    for inter_list in intersected_lines:
        intersect = []
        axis_line = inter_list[0]
        for second_line in inter_list:
            if second_line == axis_line:
                continue
            if type(line_intersecting(second_line, axis_line)) == bool:
                continue
            intersect = line_intersecting(second_line, axis_line)
        if intersect:
            intersect[2] = height
            if abs(round(intersect[0]) - intersect[0]) < rounding_error:
                intersect[0] = float(round(intersect[0]))
            if abs(round(intersect[1]) - intersect[1]) < rounding_error:
                intersect[1] = float(round(intersect[1]))
            if intersect not in Points:
                Points.append(intersect)
            if intersect not in roof_points:
                roof_points.append(intersect)
            #plt.scatter(intersect[0], intersect[1], c='r')
            for second_line in inter_list:
                if second_line == axis_line:
                    continue
                for second_list in intersected_lines:
                    if second_list[0] == second_line:
                        intersected_lines.remove(second_list)
                        break
                index = axis_lines.index(second_line)
                second_line.point1 = second_line.beginning
                second_line.point2 = intersect
                second_line.final = True
                #draw_line(second_line, 'r')
                axis_lines[index] = second_line
            index = axis_lines.index(axis_line)
            axis_line.point1 = axis_line.beginning
            axis_line.point2 = intersect
            axis_line.final = True
            axis_lines[index] = axis_line
            #draw_line(axis_line, 'r')
    return axis_lines


def are_closest(angle_line: Line, cool_line: Line) -> bool:
    """
    Checks if intersect of the angle line and the cool line are the closest intersect of cool line in relation to other

    lines that are not borderlines.
    """
    global Points, Lines
    if not line_intersecting(angle_line, cool_line):
        return False
    beginning_point = cool_line.beginning
    if Points.index(beginning_point) == 0:
        corner_points = [Points[1], Points[len(Points) - 1]]
    elif Points.index(beginning_point) == len(Points) - 1:
        corner_points = [Points[0], Points[len(Points) - 2]]
    else:
        corner_points = [Points[Points.index(beginning_point) - 1], Points[Points.index(beginning_point) + 1]]
    for corner_point in corner_points:
        if corner_point == angle_line.beginning:
            corner_points.remove(corner_point)
    for corner_point in corner_points:
        for second_line in Lines:
            if second_line.beginning == corner_point and not second_line.border:
                if not line_intersecting(second_line, cool_line):
                    return True
                return distance(line_intersecting(second_line, cool_line), cool_line.beginning) > \
                    distance(line_intersecting(angle_line, cool_line), cool_line.beginning)


def draw_line(aline: Line, color):
    plt.plot([aline.point1[0], aline.point2[0]], [aline.point1[1], aline.point2[1]], c=color, ls='-', lw=1.5, alpha=0.5)


def find_unfinished_lines():
    global Lines
    un_lines = []
    for unfinished_line in Lines:
        if not unfinished_line.final and not unfinished_line.border:
            un_lines.append(unfinished_line)
    return un_lines


def clear_doubles(points: list) -> list:
    for doubled_point in points:
        while points.count(doubled_point) > 1:
            points.remove(doubled_point)

    return points


def check_on_line(a_line: Line, a_point: list) -> bool:
    global rounding_error
    a = a_line.slope()
    if a >= 999999999999999:
        if abs(a_line.point1[0] - a_point[0]) <= 2 * rounding_error:
            return True
        return False
    b = a_line.point1[1] - (a_line.slope() * a_line.point1[0])
    if abs((a_point[0] * a + b) - a_point[1]) <= rounding_error:
        return True
    return False


def connect_roof_points():
    global roof_points, Points, Lines, height, unfinished_lines, used_lines, angle, animation_points, animation_lines,\
        animation_axes
    if len(roof_points) == 2 and len(unfinished_lines) == 0:
        refresh_encloseure()
        Lines.append(Line(roof_points[0], roof_points[1], roof_points[0], False, True, [0, 0]))
        #draw_line(Line(roof_points[0], roof_points[1], roof_points[0], False, True, [0, 0]), 'r')
        if roof_points[0] not in Points:
            Points.append(roof_points[0])
        if roof_points[1] not in Points:
            Points.append(roof_points[1])
        return None
    roof_points = clear_doubles(roof_points)
    roof_lines = []
    for roof_point in roof_points:
        refresh_encloseure()
        connecting_lines = []
        all_walls = []
        walls = []
        for connecting_line in Lines:
            if connecting_line.point1 == roof_point or connecting_line.point2 == roof_point:
                connecting_lines.append(connecting_line)
        for con_line in connecting_lines:
            all_walls.append(con_line.walls[0])
            all_walls.append(con_line.walls[1])
        for number in all_walls:
            if all_walls.count(number) == 1:
                walls.append(number)
        border_line1 = Lines[walls[0] - 1]
        border_line2 = Lines[walls[1] - 1]

        if type(line_intersecting(border_line1, border_line2)) == bool:
            roof_line = Line(roof_point,
                             [border_line1.point1[0] - border_line1.point2[0] + roof_point[0], border_line1.point1[1]
                              - border_line1.point2[1] + roof_point[1], roof_point[2]], roof_point, False, False, walls)
        else:
            if distance(border_line2.point1, line_intersecting(border_line1, border_line2)) > rounding_error:
                point2 = border_line2.point1
            else:
                point2 = border_line2.point2
            roof_line = find_angle(roof_point, line_intersecting(border_line1, border_line2), point2)
            roof_line.beginning = roof_point
            roof_line.point1 = roof_point
            roof_line.point2 = line_intersecting(border_line1, border_line2)
            roof_line.walls = walls
        roof_lines.append(roof_line)

    animation_points.append([])
    for add_point in Points:
        animation_points[len(animation_points)-1].append(add_point)
    animation_axes.append([])
    for add_axis in roof_lines:
        animation_axes[len(animation_axes)-1].append(Line(add_axis.point1, add_axis.point2, add_axis.beginning, add_axis.border, add_axis.final, add_axis.walls))

    animation_lines.append([])
    for add_line in Lines:
        animation_lines[len(animation_lines)-1].append(Line(add_line.point1, add_line.point2, add_line.beginning, add_line.border, add_line.final, add_line.walls))

    for roof_line in roof_lines:
        if roof_line.final:
            continue
        #draw_line(roof_line, 'b')
        intersected_lines = []
        remove_lines = []
        intersected_points = []

        for intersected_point in Points:
            if not check_on_line(roof_line, intersected_point):
                continue
            if check_crossing(roof_line.beginning, intersected_point):
                continue
            if intersected_point == roof_line.point1 or intersected_point == roof_line.point2:
                continue
            intersected_points.append(intersected_point)
        dist = 0
        if len(intersected_points) != 0:
            dist = distance(roof_line.beginning, intersected_points[0])
        for intersected_point in intersected_points:
            if distance(roof_line.beginning, intersected_point) <= dist + rounding_error:
                dist = distance(roof_line.beginning, intersected_point)
        remove_points = []
        for intersected_point in intersected_points:
            if distance(roof_line.beginning, intersected_point) >= dist + rounding_error:
                remove_points.append(intersected_point)

        for intersect_line in Lines:
            if intersect_line.final or not line_intersecting(roof_line, intersect_line):
                continue
            if intersect_line.border and not intersect_line.banned:
                continue
            if intersect_line.border and intersect_line.banned:
                intersected_lines.append(intersect_line)
                continue
            if type(line_intersecting(roof_line, intersect_line)) == bool:
                intersected_lines.append(intersect_line)
                continue
            if check_crossing(roof_line.beginning, line_intersecting(intersect_line, roof_line)):
                continue
            if check_crossing(intersect_line.beginning, line_intersecting(intersect_line, roof_line)):
                continue
            if not check_in_shape(line_intersecting(intersect_line, roof_line)):
                continue
            intersected_lines.append(intersect_line)
        lenght = 0
        for second_roof_line in roof_lines:
            if second_roof_line.final or second_roof_line.border or not line_intersecting(roof_line,
                                                                                          second_roof_line) or \
                    second_roof_line == roof_line:
                continue
            if type(line_intersecting(roof_line, second_roof_line)) == bool:
                intersected_lines.append(second_roof_line)
                continue
            if check_crossing(roof_line.beginning, line_intersecting(second_roof_line, roof_line)):
                continue
            if check_crossing(second_roof_line.beginning, line_intersecting(second_roof_line, roof_line)):
                continue
            if check_all_crossings(roof_line, second_roof_line, roof_line.beginning):
                continue
            if check_all_crossings(second_roof_line, roof_line, second_roof_line.beginning):
                continue
            if not check_in_shape(line_intersecting(second_roof_line, roof_line)):
                continue
            if not check_on_line(roof_line, line_intersecting(roof_line, second_roof_line)):
                continue
            intersected_lines.append(second_roof_line)
        for intersect_line in intersected_lines:
            if type(line_intersecting(roof_line, intersect_line)) == bool:
                continue
            lenght = distance(roof_line.beginning, line_intersecting(intersect_line, roof_line))
            break
        intersects = []
        for intersect_line in intersected_lines:
            if type(line_intersecting(roof_line, intersect_line)) == bool:
                continue
            if distance(roof_line.beginning, line_intersecting(intersect_line, roof_line)) <= lenght + rounding_error:
                lenght = distance(roof_line.beginning, line_intersecting(intersect_line, roof_line))
        for intersect_line in intersected_lines:
            if type(line_intersecting(roof_line, intersect_line)) == bool:
                continue
            if distance(roof_line.beginning, line_intersecting(intersect_line, roof_line)) >= lenght + rounding_error:
                remove_lines.append(intersect_line)
        for remove_point in remove_points:
            intersected_points.remove(remove_point)
        for remove_line in remove_lines:
            intersected_lines.remove(remove_line)

        for intersect_line in intersected_lines:
            if type(line_intersecting(roof_line, intersect_line)) == bool:
                continue
            intersect = line_intersecting(intersect_line, roof_line)
            if intersect not in intersects:
                intersects.append(intersect)
        ban = False
        if (dist < lenght or lenght == 0) and dist != 0:
            roof_line.point2 = intersected_points[0]
            roof_line.point1 = roof_line.beginning
            roof_line.final = True
            #draw_line(roof_line, 'r')
            Lines.append(roof_line)
        if len(intersected_points) == 0 and len(intersected_lines) == 0:
            continue
        else:
            for intersect_line in intersected_lines:
                if intersect_line.banned:
                    ban = True
            if ban:
                intersect_line = intersected_lines[0]
                intersect[2] = height
                if abs(round(intersect[0]) - intersect[0]) < rounding_error:
                    intersect[0] = float(round(intersect[0]))
                if abs(round(intersect[1]) - intersect[1]) < rounding_error:
                    intersect[1] = float(round(intersect[1]))
                if intersect not in Points:
                    Points.append(intersect)
                #plt.scatter(intersect[0], intersect[1], c='r')
                if walls[0] < walls[1]:
                    border_line1 = Lines[walls[0] - 1]
                    border_line2 = Lines[walls[1] - 1]
                else:
                    border_line1 = Lines[walls[1] - 1]
                    border_line2 = Lines[walls[0] - 1]
                if walls[0] < walls[1]:
                    con_line1 = Line(intersect, intersect_line.point1, intersect_line.point1, False, True,
                                     [border_line1.walls[0], intersect_line.walls[0]])
                    con_line2 = Line(intersect, intersect_line.point2, intersect_line.point2, False, True,
                                     [border_line2.walls[0], intersect_line.walls[0]])
                else:
                    con_line2 = Line(intersect, intersect_line.point1, intersect_line.point1, False, True,
                                     [border_line1.walls[0], intersect_line.walls[0]])
                    con_line1 = Line(intersect, intersect_line.point2, intersect_line.point2, False, True,
                                     [border_line2.walls[0], intersect_line.walls[0]])
                #draw_line(con_line2, 'r')
                #draw_line(con_line1, 'r')
                roof_line.point2 = intersect
                roof_line.point1 = roof_line.beginning
                roof_line.final = True
                roof_points.remove(roof_line.beginning)
                #draw_line(roof_line, 'r')
                Lines.append(con_line1)
                Lines.append(con_line2)
                Lines.append(roof_line)
            else:
                if not check_on_line(roof_line, intersect):
                    continue
                intersect[2] = height
                if abs(round(intersect[0]) - intersect[0]) < rounding_error:
                    intersect[0] = float(round(intersect[0]))
                if abs(round(intersect[1]) - intersect[1]) < rounding_error:
                    intersect[1] = float(round(intersect[1]))
                if intersect not in Points:
                    Points.append(intersect)
                #plt.scatter(intersect[0], intersect[1], c='r')
                roof_line.point2 = intersect
                roof_line.point1 = roof_line.beginning
                roof_line.final = True
                #draw_line(roof_line, 'r')
                roof_points.append(intersect)
                roof_points.remove(roof_line.beginning)
                if roof_line not in Lines:
                    Lines.append(roof_line)
                for interline in intersected_lines:
                    interline.point1 = interline.beginning
                    interline.point2 = intersect
                    interline.final = True
                    if interline not in Lines:
                        Lines.append(interline)
                    #draw_line(interline, 'r')

    unfinished_lines = find_unfinished_lines()
    refresh_encloseure()
    roof_points = check_finished(roof_points)
    if len(roof_points) == 1 and len(unfinished_lines) == 0:
        return
    if len(unfinished_lines) != 0 or len(roof_points) != 0:  #
        connect_roof_points()



def check_finished(possible_points: list[list]):
    global Lines, Points
    keep_points = []
    for possible_point in possible_points:
        walls = []
        for con_line in Lines:
            if con_line.point1 == possible_point or con_line.point2 == possible_point:
                walls.append(con_line.walls[0])
                walls.append(con_line.walls[1])
        for wall in walls:
            if walls.count(wall) != 2:
                if possible_point not in keep_points:
                    keep_points.append(possible_point)
                    break
    for wrong_point in possible_points:
        if wrong_point not in keep_points and wrong_point not in Points:
            Points.append(wrong_point)
    return keep_points


def check_all_crossings(first_line: Line, second_line: Line, first_line_beginning: list) -> bool:
    global Lines, rounding_error
    first_point = line_intersecting(first_line, second_line)
    second_point = first_line_beginning
    temp_line = Line(first_point, second_point, second_point, False, False, [0, 0])
    for tried_line in Lines:
        if tried_line.border:
            continue
        if tried_line.final:
            continue
        if tried_line == first_line or tried_line == second_line:
            continue
        if check_on_line(tried_line, first_point) or check_on_line(tried_line, second_point):
            continue
        if type(line_intersecting(tried_line, temp_line)) != bool:
            intersect = line_intersecting(tried_line, temp_line)
            if abs(distance(intersect, first_point) + distance(intersect, second_point) -
                   distance(first_point, second_point)) <= rounding_error:
                return True
    return False


def check_in_shape(check_point: list) -> bool:
    crossed_lines = []
    point_line = Line(check_point, [check_point[0] + 10, check_point[1], check_point[2]], check_point, False, False,
                      [0, 0])
    for crossed_line in Lines:
        if line_intersecting(point_line, crossed_line) and crossed_line.border:
            if type(line_intersecting(point_line, crossed_line)) == bool:
                crossed_lines.append(crossed_line)
                continue
            if line_intersecting(point_line, crossed_line)[0] < check_point[0]:
                continue
            if crossed_line.point1[0] == crossed_line.point2[0]:
                if crossed_line.point1[1] - rounding_error <= line_intersecting(point_line, crossed_line)[1] <= \
                        crossed_line.point2[1] or crossed_line.point1[1] - rounding_error >= \
                        line_intersecting(point_line, crossed_line)[1] >= crossed_line.point2[1]:
                    crossed_lines.append(crossed_line)
            elif crossed_line.point1[0] <= crossed_line.point2[0]:
                if crossed_line.point1[0] - rounding_error <= line_intersecting(point_line, crossed_line)[0] <= \
                        crossed_line.point2[0]:
                    crossed_lines.append(crossed_line)
            elif crossed_line.point1[0] >= crossed_line.point2[0]:
                if crossed_line.point1[0] - rounding_error >= line_intersecting(point_line, crossed_line)[0] >= \
                        crossed_line.point2[0]:
                    crossed_lines.append(crossed_line)

    if len(crossed_lines) % 2 == 0:
        return False
    return True


def refresh_encloseure():
    global Lines, Points
    for refresh_line in Lines:
        if refresh_line.border and not refresh_line.final:
            refresh_line.final = check_enclosed(refresh_line)
        if not refresh_line.border:
            if refresh_line.point1 in Points and refresh_line.point2 in Points:
                refresh_line.final = True
    for refresh_point in Points:
        for dupli_point in Points:
            if refresh_point[0] == dupli_point[0] and refresh_point[1] == dupli_point[1] and dupli_point[2] != \
                    refresh_point[2]:
                Points.remove(dupli_point)
            elif abs(refresh_point[0] - dupli_point[0]) < rounding_error and abs(
                    refresh_point[1] - dupli_point[1]) < rounding_error and dupli_point != refresh_point:
                for conf_line in Lines:
                    if conf_line.point1 == dupli_point:
                        conf_line.point1 = refresh_point
                    if conf_line.point2 == dupli_point:
                        conf_line.point2 = refresh_point
                    if conf_line.beginning == dupli_point:
                        conf_line.beginning = refresh_point
                Points.remove(dupli_point)
    for refresh_line in Lines:
        if refresh_line.point1 == refresh_line.point2:
            Lines.remove(refresh_line)
        for good_point in Points:
            if refresh_line.point1[1] == good_point[1] and refresh_line.point1[0] == good_point[0] and \
                    refresh_line.point1[2] != good_point[2]:
                refresh_line.point1 = good_point
            if refresh_line.point2[1] == good_point[1] and refresh_line.point2[0] == good_point[0] and \
                    refresh_line.point2[2] != good_point[2]:
                refresh_line.point2 = good_point
            if refresh_line.beginning[1] == good_point[1] and refresh_line.beginning[0] == good_point[0] and \
                    refresh_line.beginning[2] != good_point[2]:
                refresh_line.beginning = good_point


def get_walls(wall1: Line, wall2: Line) -> list:
    walls = []
    walls1 = wall1.walls
    walls2 = wall2.walls
    if walls2.count(walls1[0]) == 0:
        walls.append(walls1[0])
    else:
        walls.append(walls1[1])
    if walls1.count(walls2[0]) == 0:
        walls.append(walls2[0])
    else:
        walls.append(walls2[1])
    return walls


def check_enclosed(border_line: Line) -> bool:
    global Lines, Points
    close_points = recursion_maze(border_line.point1, border_line.point2, [border_line.point1], 0, False,
                                  border_line.walls[0])
    if not close_points:
        return False
    close_points.append(border_line.point2)
    for first_point in close_points:
        if close_points.index(first_point) == len(close_points) - 1:
            second_point = close_points[0]
        else:
            second_point = close_points[close_points.index(first_point) + 1]
        for around_line in Lines:
            if (around_line.point1 == first_point or around_line.point2 == first_point) and (
                    around_line.point1 == second_point or around_line.point2 == second_point):
                if around_line == border_line:
                    continue
                if around_line.border:
                    return False
    for enclose_point in close_points:
        close_points[close_points.index(enclose_point)] = Points.index(enclose_point) + len(worthno_points)
    border_line.enclose_points = close_points
    return True


def shorten_line(long_line: Line, line_point: list):
    if (math.sqrt(math.pow((line_point[0] - long_line.point1[0]), 2) + math.pow(line_point[1] - long_line.point1[1], 2))
            < math.sqrt(
                math.pow((line_point[0] - long_line.point1[0]), 2) + math.pow(line_point[1] - long_line.point1[1], 2))):
        long_line.point2 = long_line.beginning
    else:
        long_line.point1 = long_line.beginning
    return long_line


def line_intersecting(line1: Line, line2: Line):
    global rounding_error, height
    a1 = line1.slope()
    a2 = line2.slope()

    if a1 >= 999999999999999:
        if a2 >= 999999999999999:
            if abs(line1.point1[0] - line2.point1[0]) <= 2 * rounding_error:
                return True
            return False

        b2 = line2.point1[1] - (line2.slope() * line2.point1[0])
        x = line1.point1[0]
        y = a2 * x + b2

    elif a2 >= 999999999999999:
        b1 = line1.point1[1] - (line1.slope() * line1.point1[0])
        x = line2.point1[0]
        y = a1 * x + b1

    else:
        b1 = line1.point1[1] - (line1.slope() * line1.point1[0])
        b2 = line2.point1[1] - (line2.slope() * line2.point1[0])
        if abs(a1 - a2) <= rounding_error:
            if abs(b1 - b2) <= rounding_error:
                return True
            return False
        x = ((b1 - b2) / (a2 - a1))
        y = a1 * x + b1

    return [x, y, height]


def new_faces():
    global Lines, Points, Faces
    true_lines = []
    for possible_line in Lines:
        if possible_line.border:
            true_lines.append(possible_line)
    for border_line in true_lines:
        if border_line.enclose_points:
            indexes = border_line.enclose_points
            for number in range(len(indexes)):
                indexes[number] = indexes[number]
        else:
            indexes = []
            recursion = list(recursion_maze(border_line.point1, border_line.point2, [border_line.point1], 0, False,
                                            border_line.walls[0]))
            recursion.append(border_line.point2)
            for shape_point in recursion:
                indexes.append(Points.index(shape_point))
        Faces.append(indexes)


def recursion_maze(start_point: list, finish_point: list, passed_points: list[list], i: int, can_use_border: bool,
                   wall_number: int):
    i = i + 1
    if i > 500:
        return False
    global Points
    if advanced_connected(start_point, finish_point, can_use_border) and start_point not in passed_points:
        passed_points.append(start_point)
        return passed_points
    connect_pointss = []
    close_point = []
    for possible_point in Points:
        if possible_point == start_point or possible_point == finish_point:
            continue
        if possible_point in passed_points:
            continue
        if advanced_connected(possible_point, start_point, can_use_border):
            connect_pointss.append(possible_point)
    for possible_point in connect_pointss:
        if not close_point:
            close_point = possible_point
            continue
        if distance(possible_point, finish_point) < distance(close_point, finish_point):
            close_point = possible_point
    if start_point not in passed_points:
        passed_points.append(start_point)
    return recursion_maze(close_point, finish_point, passed_points, i, can_use_border, wall_number)


def connected(point1: list, point2: list):
    global Lines
    for connect_line in Lines:
        if connect_line.point1 == point1 or connect_line.point1 == point2:
            if connect_line.point2 == point1 or connect_line.point2 == point2:
                return True
    return False


def advanced_connected(point1: list, point2: list, can_border_lines: bool):
    global Lines
    for connect_line in Lines:
        if not can_border_lines:
            if connect_line.border:
                continue
        if (connect_line.point1 == point1 and connect_line.point2 == point2) or (connect_line.point2 == point1 and
                                                                                 connect_line.point1 == point2):
            return True
    return False


def fix_points():
    global Points, Lines, height, angle
    for roof_point in Points:
        border = False
        connected_points = []
        for border_line in Lines:
            if border_line.border and (border_line.point1 == roof_point or border_line.point2 == roof_point):
                border = True
        if border:
            continue
        for border_point in Points:
            if connected(border_point, roof_point):
                connected_points.append(border_point)
        for connected_point in connected_points:
            for border_line in Lines:
                if border_line.border and (
                        border_line.point1 == connected_point or border_line.point2 == connected_point):
                    final_point = connected_point

        roof_point[2] = math.tan(angle) * distance(roof_point, final_point) + final_point[2]


def distance(point1: list, point2: list):
    return math.sqrt(math.pow((point1[0] - point2[0]), 2) + math.pow(point1[1] - point2[1], 2))


def rebuild_points():
    global worthno_points, Points
    for roof_point in Points:
        worthno_points.append(roof_point)
    Points = worthno_points


def fine_make_a_file():
    global Points, Faces
    file_string = """CubePoints = [\n"""
    for final_point in Points:
        file_string = file_string + str(final_point)
        if Points.index(final_point) != len(Points) - 1:
            file_string = file_string + ",\n"
        else:
            file_string = file_string + "];\n"
    file_string = file_string + "\n"
    file_string = file_string + "CubeFaces = [\n"
    for face in Faces:
        file_string = file_string + str(face)
        if Faces.index(face) != len(Faces) - 1:
            file_string = file_string + ",\n"
        else:
            file_string = file_string + "];\n"
    file_string = file_string + "\n"
    file_string = file_string + "polyhedron( CubePoints, CubeFaces );"
    file_to_write = open("house.scad", 'w')
    file_to_write.write(file_string)
    file_to_write.close()
    if printing:
        print(file_string)


def onClick(event):
    global unpause
    unpause ^= True


def update(frame: int):
    global animation_points, animation_lines, animation_axes, border_points, unpause, number_of_frames
    if keyboard.is_pressed(' '):
        number_of_frames += 1
        number_of_frames = number_of_frames % len(animation_points)
        print(number_of_frames)
        ax.clear()
        change_drawing(ax)
        for draw_point in animation_points[number_of_frames]:
            if draw_point in border_points:
                plt.scatter(draw_point[0], draw_point[1], c='m')
            else:
                plt.scatter(draw_point[0], draw_point[1], c='r')
        for drawn_line in animation_lines[number_of_frames]:
            if drawn_line.border:
                draw_line(drawn_line, 'm')
            elif drawn_line.final:
                draw_line(drawn_line, 'r')
            else:
                draw_line(drawn_line, 'y')
        for drawn_axis in animation_axes[number_of_frames]:
            draw_line(drawn_axis, 'b')


if __name__ == '__main__':
    if f.find("roof_height") < 0:
        Points = list(find_points(f))
        Faces = list(find_faces(f))
        Faces = list(not_top_faces(Points, Faces))
    else:
        Points = list(find_points(f))
        Faces = list(find_faces(f))
        height = find_height(f)
        create_faces(Points)
        top_points = list(add_top_points(Points, height))
        for top_point in top_points:
            Points.append(top_point)
    if f.find("roof_angle") > 0:
        angle = find_roof_angle(f)

    worthno_points = worthless_points(Points)
    Lines = list(border_lines(Points))

    # used for drawing in steps

    for point in Points:
        border_points.append(point)

    if f.find("banned_walls") > 0:
        find_banned_walls(f)

    start_drawing()
    draw(Points)

    # This is where the actual math begins

    setup_angles(Points)
    find_lines()
    unfinished_lines = find_unfinished_lines()

    connect_roof_points()
    animation_lines.append(Lines)
    animation_points.append(Points)
    animation_axes.append([])
    print(len(animation_lines))

    refresh_encloseure()

    fix_points()
    rebuild_points()

    new_faces()

    fine_make_a_file()
    fig.canvas.mpl_connect('button_press_event', onClick)
    anim = animation.FuncAnimation(fig=fig, func=update, frames=len(animation_points), interval=100)

    plt.show()

