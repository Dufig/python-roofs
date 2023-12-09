import numpy as np  # Only used for graphing
import matplotlib.pyplot as plt  # Only used for graphing
import math

HEIGHT = 0
f = open("S_house.scad", "r")
f = f.read()
DRAWING = True
PRINTING = True
roof_points = []
ROUNDING_ERROR = 0.00005
ANGLE = math.radians(45)


class Line:
    def __init__(self, point1: list, point2: list, beginning: list, border: bool, final: bool, walls: list[int]):
        self.point1 = point1
        self.point2 = point2
        self.beginning = beginning
        self.border = border
        self.final = final
        self.walls = walls
        self.enclose_points = []

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
               + str(self.walls)

    def lenght(self):
        return math.sqrt(math.pow((self.point1[0] - self.point2[0]), 2) + math.pow(self.point1[1] - self.point2[1], 2))


def draw(points: list):
    # Enter x and y coordinates of points and colors
    xs = []
    ys = []
    colors = ['m']
    for toppp_point in points:
        xs.append(toppp_point[0])
        ys.append(toppp_point[1])

    # Select length of axes and the space between tick labels
    xmin, xmax, ymin, ymax = -5, 40, -5, 40
    ticks_frequency = 5

    # Plot points
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.scatter(xs, ys, c=colors)

    for line_point in points:
        if line_point == points[0]:
            ax.plot([line_point[0], points[points.index(line_point) + 1][0]], [line_point[1],
                                                                               points[points.index(line_point) + 1][1]],
                    c='m')
        elif line_point == points[len(points) - 1]:
            ax.plot([line_point[0], points[0][0]], [line_point[1], points[0][1]], c='m')
        else:
            ax.plot([line_point[0], points[points.index(line_point) + 1][0]], [line_point[1],
                                                                               points[points.index(line_point) + 1][1]],
                    c='m')

    # Set identical scales for both axes
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


'''
Both of these functions are just to find the points and the faces respectively, .scad is a pretty rare format to work 

with in python so I didn't find a library to do this.
'''


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
    global HEIGHT
    if no_list.count("roof_height") != 1:
        return None
    no_list = no_list[no_list.index("roof_height"):]
    no_list = no_list[no_list.index("=") + 1:]
    no_list = no_list[:no_list.index(";")]
    no_list = float(no_list)
    return no_list


def find_roof_angle(no_list: str):
    global ANGLE
    if no_list.count("roof_angle") != 1:
        return None
    no_list = no_list[no_list.index("roof_angle"):]
    no_list = no_list[no_list.index("=") + 1:]
    no_list = no_list[:no_list.index(";")]
    no_list = float(no_list)
    no_list = math.radians(no_list)
    return no_list


def not_top_faces(points: list, faces: list) -> list:
    # Gets rid of the top face, as it will get covered by the roof
    global HEIGHT, f
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
        HEIGHT = heights[len(heights) - 1] + (heights[len(heights) - 1] - heights[0]) * find_height(f)
    else:
        HEIGHT = heights[len(heights) - 1] + heights[len(heights) - 1] - heights[0]
    return good_faces


def worthless_points(points: list) -> list:
    # Dump points that don't appear in the top layer
    heights = []
    bad_points = []
    for bad_point in points:
        if bad_point[2] not in heights:
            heights.append(bad_point[2])
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
        for bottom_point in low_points:
            face.append(low_points.index(bottom_point))
        Faces = [face]
    if HEIGHT == 0:
        return None
    for bottom_point in low_points:
        if low_points.index(bottom_point) == len(low_points) - 1:
            face = [low_points.index(bottom_point), 2 * len(low_points) - 1, len(low_points), 0]
        else:
            face = [low_points.index(bottom_point), len(low_points) + low_points.index(bottom_point), len(low_points) +
                    low_points.index(bottom_point) + 1, low_points.index(bottom_point) + 1]
        Faces.append(face)


def find_angle(a: list, mid: list, c: list):
    """Finds the angle axis by converting the two lines making up the angle into vectors, making them the same length

    and adding them up. The resulting line goes the same distance both ways from its beginning.
    """
    global HEIGHT
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
    return Line([xc + xa + xmid, ya + yc + ymid, HEIGHT], [- xc - xa + xmid, - ya - yc + ymid, HEIGHT],
                [xmid, ymid, zmid], False, False, [0, 0])


def border_lines(points: list) -> list:
    lines = []
    for border_point in points:
        walls = [Points.index(border_point) + 1, Points.index(border_point) + 1]
        if points.index(border_point) == len(points) - 1:
            lines.append(Line(border_point, Points[0], border_point, True, False, walls))
        else:
            lines.append(Line(border_point, Points[Points.index(border_point) + 1], border_point, True, False, walls))

    return lines


def setup_angles(points: list):
    # Just a setup for the find_angle function.
    for angle_point in points:
        wall_lines = []
        if points.index(angle_point) == 0:
            angle_line = find_angle(points[len(points) - 1], points[0], points[1])
        elif points.index(angle_point) == len(points) - 1:
            angle_line = find_angle(points[len(points) - 2], points[len(points) - 1], points[0])
        else:
            angle_line = find_angle(points[points.index(angle_point) - 1], points[points.index(angle_point)],
                                    points[points.index(angle_point) + 1])
        for wall_line in Lines:
            if wall_line.border and (wall_line.point1 == angle_point or wall_line.point2 == angle_point):
                wall_lines.append(wall_line)
        if len(wall_lines) == 2:
            walls = get_walls(wall_lines[0], wall_lines[1])
            angle_line.walls = walls
        Lines.append(angle_line)


def check_crossing(point1: list[int], point2: list[int]) -> bool:
    global Lines, ROUNDING_ERROR
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
        if not line_intersecting(temp_line, border_line):
            continue
        intersect = line_intersecting(temp_line, border_line)
        if abs(distance(border_line.point1, border_line.point2) - (distance(border_line.point1, intersect) +
                                                                   distance(border_line.point2,
                                                                            intersect))) > 2 * ROUNDING_ERROR:
            continue
        if abs(distance(point1, point2) - (distance(point1, intersect) + distance(point2, intersect))) < \
                2 * ROUNDING_ERROR:
            return True
    return False


def find_lines():
    global Points, Lines, roof_points

    axis_lines = []
    for unborder_line in Lines:
        if not unborder_line.border:
            axis_lines.append(unborder_line)
    intersects = []
    for axis_line in axis_lines:
        other_lines = [axis_line]
        for second_line in axis_lines:
            if second_line == axis_line:
                continue
            len_a = 99999999999

            if line_intersecting(second_line, axis_line):
                if check_crossing(axis_line.beginning, line_intersecting(second_line, axis_line)):
                    continue
                if check_crossing(second_line.beginning, line_intersecting(second_line, axis_line)):
                    continue
                if not check_in_shape(line_intersecting(second_line, axis_line)):
                    continue
                other_lines.append(second_line)
                if distance(line_intersecting(second_line, axis_line), axis_line.beginning) < len_a:
                    len_a = distance(line_intersecting(second_line, axis_line), axis_line.beginning)
        intersects.append(other_lines)
    for axis_line in axis_lines:
        if axis_line.final:
            continue
        for list in intersects:
            if list[0] == axis_line:
                lines = list
        if len(lines) == 1:
            continue
        possible_lines = []
        remove_lines = []
        len_a = 0
        for second_line in lines:
            if second_line == axis_line:
                continue
            if distance(line_intersecting(second_line, axis_line), axis_line.beginning) > len_a:
                len_a = distance(line_intersecting(second_line, axis_line), axis_line.beginning)
        for second_line in lines:
            if second_line == axis_line or second_line.final == True:
                continue
            len_two = distance(line_intersecting(second_line, axis_line), second_line.beginning)
            for a_list in intersects:
                if a_list[0] == second_line:
                    the_list = a_list
            for third_line in the_list:
                if third_line == second_line or third_line == axis_line or third_line.final:
                    continue
                if distance(line_intersecting(second_line, third_line), second_line.beginning) < len_two:
                    if second_line not in remove_lines:
                        remove_lines.append(second_line)
        for second_line in lines:
            if second_line == axis_line or second_line in remove_lines or second_line.final:
                continue
            if abs(distance(line_intersecting(second_line, axis_line), axis_line.beginning)) < len_a + ROUNDING_ERROR:
                possible_lines = [second_line]
                len_a = distance(line_intersecting(second_line, axis_line), axis_line.beginning)
        if not possible_lines:
            continue
        intersect = line_intersecting(axis_line, possible_lines[0])
        if abs(round(intersect[0]) - intersect[0]) < ROUNDING_ERROR:
            intersect[0] = float(round(intersect[0]))
        if abs(round(intersect[1]) - intersect[1]) < ROUNDING_ERROR:
            intersect[1] = float(round(intersect[1]))
        intersect[2] = HEIGHT

        for copy_list in intersects:
            if axis_line in copy_list:
                copy_list[copy_list.index(axis_line)].final = True
            for second_line in possible_lines:
                if second_line in copy_list:
                    copy_list[copy_list.index(second_line)].final = True

        axis_line.point1 = axis_line.beginning
        axis_line.point2 = intersect
        axis_line.final = True
        draw_line(axis_line, 'r')
        for second_line in possible_lines:
            for second_list in intersects:
                if second_list[0] == second_line:
                    intersects[intersects.index(second_list)] = [second_line]
            second_line.point1 = second_line.beginning
            second_line.point2 = intersect
            second_line.final = True
            draw_line(second_line, 'r')
        roof_points.append(intersect)
        Points.append(intersect)
        plt.scatter(intersect[0], intersect[1], c='r')


def are_closest(angle_line: Line, cool_line: Line) -> bool:
    """
    Checks if intersect of the angle line and the cool line are the closest intersect of cool line in relation to other

    lines that are not borderlines.
    """
    global Points, Lines
    if not line_intersecting(angle_line, cool_line):
        return False
    first_point = cool_line.beginning
    if Points.index(first_point) == 0:
        corner_points = [Points[1], Points[len(Points) - 1]]
    elif Points.index(first_point) == len(Points) - 1:
        corner_points = [Points[0], Points[len(Points) - 2]]
    else:
        corner_points = [Points[Points.index(first_point) - 1], Points[Points.index(first_point) + 1]]
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


def connect_roof_points():
    global roof_points, Points, Lines, HEIGHT, unfinished_lines, ANGLE
    unfinished_border = 0
    for borde_line in Lines:
        if borde_line.border and not borde_line.final:
            unfinished_border = unfinished_border + 1

    if len(roof_points) <= 2 and unfinished_border <= 2 and len(unfinished_lines) <= 2:
        if len(roof_points) == 1 and unfinished_border <= 0 and len(unfinished_lines) <= 0:
            return None
        refresh_encloseure()
        Lines.append(Line(roof_points[0], roof_points[1], roof_points[0], False, True, [0, 0]))
        draw_line(Line(roof_points[0], roof_points[1], roof_points[0], False, True, [0, 0]), 'r')
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
        walls = []

        for connecting_line in Lines:
            if connecting_line.point1 == roof_point or connecting_line.point2 == roof_point:
                connecting_lines.append(connecting_line)
        if len(connecting_lines) != 2:
            temp_walls = []
            for connecting_line in connecting_lines:
                temp_walls.append(connecting_line.walls[0])
                temp_walls.append(connecting_line.walls[1])
            for number in temp_walls:
                if temp_walls.count(number) == 1:
                    walls.append(number)
        if len(connecting_lines) == 2:
            walls = get_walls(connecting_lines[0], connecting_lines[1])
        border_line1 = Lines[walls[0] - 1]
        border_line2 = Lines[walls[1] - 1]
        if line_intersecting(border_line1, border_line2):
            roof_line = Line(roof_point, line_intersecting(border_line1, border_line2), roof_point, False, False, walls)
        else:
            roof_line = Line(roof_point, [border_line1.point1[0] - border_line1.point2[0] + roof_point[0],
                                          border_line1.point1[1] - border_line1.point2[1] + roof_point[1],
                                          roof_point[2]], roof_point, False, False, walls)
        roof_lines.append(roof_line)


    roof_walls = []
    for roof_line in roof_lines:
        if roof_line.walls in roof_walls:
            roof_lines.remove(roof_line)
        roof_walls.append(roof_line.walls)
    sets = []
    for walls in roof_walls:
        temp_list = [walls]
        cont = False
        for set in sets:
            if set.count(walls) > 0:
                cont = True
        if cont:
            continue
        for second in roof_walls:
            cont = False
            if walls == second:
                continue
            if walls[0] == second[0] or walls[0] == second[1] or walls[1] == second[0] or walls[1] == second[1]:
                for set in sets:
                    if set.count(second) > 0:
                        cont = True
            if cont:
                continue
            temp_list.append(second)
        sets.append(temp_list)
    for set in sets:
        walls1 = set[0]
        walls2 = set[1] #there can't be more than 2 in a set
        use_3 = False
        is_3 = False
        for possible_line in roof_lines:
            if possible_line.walls == walls1:
                line1 = possible_line
            if possible_line.walls == walls2:
                line2 = possible_line
        if line_intersecting(line1, line2):
            additional_walls = []
            for wall_number in [walls1[0], walls1[1], walls2[0], walls2[1]]:
                if wall_number in additional_walls:
                    additional_walls.remove(wall_number)
                else:
                    additional_walls.append(wall_number)
            for axis_line in Lines:
                if axis_line.walls == additional_walls or (axis_line.walls[0] == additional_walls[1] and\
                        axis_line.walls[1] == additional_walls[0]):
                    line3 = axis_line
                    is_3 = True
            if is_3:
                if line_intersecting(line1, line3) and line_intersecting(line2, line3):
                    if distance(line_intersecting(line1, line3), line_intersecting(line1, line2)) +\
                       distance(line_intersecting(line2, line3), line_intersecting(line1, line2)) <= 3 * ROUNDING_ERROR:
                       use_3 = True
            intersect = line_intersecting(line1, line2)
            if not check_in_shape(intersect):
                continue

            intersect[2] = HEIGHT
            if abs(round(intersect[0]) - intersect[0]) < ROUNDING_ERROR:
                intersect[0] = float(round(intersect[0]))
            if abs(round(intersect[1]) - intersect[1]) < ROUNDING_ERROR:
                intersect[1] = float(round(intersect[1]))
            if intersect not in Points:
                Points.append(intersect)
            line1.point1 = line1.beginning
            line1.point2 = intersect
            line1.final = True
            line2.point2 = intersect
            line2.point1 = line2.beginning
            line2.final = True
            roof_points.append(intersect)
            draw_line(line1, 'r')
            draw_line(line2, 'r')
            plt.scatter(intersect[0], intersect[1], c='r')
            if use_3:
                line3.point2 = intersect
                line3.point1 = line3.beginning
                line3.final = True
                draw_line(line3, 'r')
            roof_points.remove(line1.beginning)
            roof_points.remove(line2.beginning)
    for roof_line in roof_lines:
        len_a = 0
        interlines = []
        for second_line in Lines:
            if not line_intersecting(roof_line, second_line) or second_line == roof_line or second_line.final or \
                    second_line.border:
                continue
            if check_crossing(second_line.beginning, line_intersecting(roof_line, second_line)):
                continue
            len_a = distance(roof_line.beginning, line_intersecting(roof_line, second_line))
            break
        for second_line in Lines:
            if not line_intersecting(roof_line, second_line) or second_line == roof_line or second_line.final or \
                    second_line.border:
                continue
            if check_crossing(second_line.beginning, line_intersecting(roof_line, second_line)):
                continue
            if distance(roof_line.beginning, line_intersecting(roof_line, second_line)) <= len_a + ROUNDING_ERROR:
                len_a = distance(roof_line.beginning, line_intersecting(roof_line, second_line))
        for second_line in Lines:
            if not line_intersecting(roof_line, second_line) or second_line == roof_line or second_line.border or \
                    second_line.final:
                continue
            if check_crossing(second_line.beginning, line_intersecting(roof_line, second_line)):
                continue
            if distance(roof_line.beginning, line_intersecting(roof_line, second_line)) <= len_a + ROUNDING_ERROR:
                interlines.append(second_line)
        if len(interlines) == 0:
            continue
        if len(interlines) == 1:
            interline = interlines[0]
            intersect = line_intersecting(roof_line, interline)
            if not check_in_shape(intersect):
                continue

            intersect[2] = HEIGHT
            if abs(round(intersect[0]) - intersect[0]) < ROUNDING_ERROR:
                intersect[0] = float(round(intersect[0]))
            if abs(round(intersect[1]) - intersect[1]) < ROUNDING_ERROR:
                intersect[1] = float(round(intersect[1]))
            if intersect not in Points:
                Points.append(intersect)
            interline.point1 = interline.beginning
            interline.point2 = intersect
            interline.final = True
            roof_line.point2 = intersect
            roof_line.point1 = roof_line.beginning
            roof_line.final = True
            roof_points.append(intersect)
            roof_points.remove(roof_line.beginning)
            Lines.append(roof_line)
            plt.scatter(intersect[0], intersect[1], c='r')
            draw_line(roof_line, 'r')
            draw_line(interline, 'r')
        else:
            intersects = []
            for interline in interlines:
                if not line_intersecting(roof_line, interline):
                    interlines.remove(interline)
                intersects.append(line_intersecting(roof_line, interline))
            first_intersect = intersects[0]
            for intersect in intersects:
                if intersect == first_intersect:
                    continue
                if abs(first_intersect[0] - intersect[0]) < ROUNDING_ERROR and abs(first_intersect[1] - intersect[1]) \
                        < ROUNDING_ERROR:
                    continue
                for interline in interlines:
                    if not line_intersecting(roof_line, interline):
                        continue
                    if line_intersecting(roof_line, interline) == intersect:
                        interlines.remove(interline)
            intersect = first_intersect
            if abs(round(intersect[0]) - intersect[0]) < ROUNDING_ERROR:
                intersect[0] = float(round(intersect[0]))
            if abs(round(intersect[1]) - intersect[1]) < ROUNDING_ERROR:
                intersect[1] = float(round(intersect[1]))
            intersect[2] = HEIGHT
            if intersect not in Points:
                Points.append(intersect)
            roof_line.point2 = intersect
            roof_line.point1 = roof_line.beginning
            roof_line.final = True
            for interline in interlines:
                interline.point1 = interline.beginning
                interline.point2 = intersect
                interline.final = True
                draw_line(interline, 'r')
            roof_points.append(intersect)
            roof_points.remove(roof_line.beginning)
            Lines.append(roof_line)
            plt.scatter(intersect[0], intersect[1], c='r')
            draw_line(roof_line, 'r')

    unfinished_lines = find_unfinished_lines()
    refresh_encloseure()
    if len(unfinished_lines) != 0 or len(roof_points) != 0:
        connect_roof_points()


def check_in_shape(checked_point: list) -> bool:
    crossed_lines = []
    point_line = Line(checked_point, [checked_point[0] + 10, checked_point[1], checked_point[2]], checked_point, False,
                      False, [0, 0])
    for crossed_line in Lines:
        if line_intersecting(point_line, crossed_line) and crossed_line.border:
            if line_intersecting(point_line, crossed_line)[0] < checked_point[0]:
                continue
            if crossed_line.point1[0] == crossed_line.point2[0]:
                if crossed_line.point1[1] - ROUNDING_ERROR <= line_intersecting(point_line, crossed_line)[1] <= \
                        crossed_line.point2[1] or crossed_line.point1[1] - ROUNDING_ERROR >= \
                        line_intersecting(point_line, crossed_line)[1] >= crossed_line.point2[1]:
                    crossed_lines.append(crossed_line)
            elif crossed_line.point1[0] <= crossed_line.point2[0]:
                if crossed_line.point1[0] - ROUNDING_ERROR <= line_intersecting(point_line, crossed_line)[0] <= \
                        crossed_line.point2[0]:
                    crossed_lines.append(crossed_line)
            elif crossed_line.point1[0] >= crossed_line.point2[0]:
                if crossed_line.point1[0] - ROUNDING_ERROR >= line_intersecting(point_line, crossed_line)[0] >= \
                        crossed_line.point2[0]:
                    crossed_lines.append(crossed_line)

    if len(crossed_lines) % 2 == 0:
        return False
    return True


def refresh_encloseure():
    global Lines, Points
    for refresh_line in Lines:
        continu = False
        if refresh_line.border and not refresh_line.final:
            for axis_line in Lines:
                if (axis_line.beginning == refresh_line.point1 or axis_line.beginning == refresh_line.point2) and not\
                    axis_line.border:
                    if not axis_line.final:
                        continu = True
            if continu:
                continue
            refresh_line.final = check_enclosed(refresh_line)
        if not refresh_line.border:
            if refresh_line.point1 in Points and refresh_line.point2 in Points:
                refresh_line.final = True
    for refresh_point in Points:
        for dupli_point in Points:
            if refresh_point[0] == dupli_point[0] and refresh_point[1] == dupli_point[1] and dupli_point[2] != \
                    refresh_point[2]:
                Points.remove(dupli_point)
            elif abs(refresh_point[0] - dupli_point[0]) < ROUNDING_ERROR and abs(
                    refresh_point[1] - dupli_point[1]) < ROUNDING_ERROR and dupli_point != refresh_point:
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
    close_points = recursion_maze(border_line.point1, border_line.point2, [border_line.point1], 0, False)
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


def shorten_line(long_line: Line, point: list):
    if (math.sqrt(math.pow((point[0] - long_line.point1[0]), 2) + math.pow(point[1] - long_line.point1[1], 2)) <
            math.sqrt(math.pow((point[0] - long_line.point1[0]), 2) + math.pow(point[1] - long_line.point1[1], 2))):
        long_line.point2 = long_line.beginning
    else:
        long_line.point1 = long_line.beginning
    return long_line


def line_intersecting(line1: Line, line2: Line):
    if line1.slope() == line2.slope():
        return False
    a1 = line1.slope()
    a2 = line2.slope()
    if line1.point1[0] != 0:
        b1 = (line1.point1[1] - (line1.slope() * line1.point1[0]))
    elif line1.point2[0] != 0:
        b1 = (line1.point2[1] - (line1.slope() * line1.point2[0]))
    else:
        return False
    if line2.point1[0] != 0:
        b2 = (line2.point1[1] - (line2.slope() * line2.point1[0]))
    elif line1.point2[0] != 0:
        b2 = (line2.point2[1] - (line2.slope() * line2.point2[0]))
    else:
        return False
    x = ((b1 - b2) / (a2 - a1))
    y = a1 * x + b1
    if abs(round(x) - x) < ROUNDING_ERROR:
        x = round(x)
    if abs(round(y) - y) < ROUNDING_ERROR:
        y = round(y)
    return [x, y, HEIGHT]


def new_faces():
    global Lines, Points, Faces
    true_lines = []
    for possible_line in Lines:
        if possible_line.border:
            true_lines.append(possible_line)
    for border_line in true_lines:
        if border_line.enclose_points != []:
            indexes = border_line.enclose_points
            for number in range(len(indexes)):
                indexes[number] = indexes[number]
        else:
            indexes = []
            recursion = list(recursion_maze(border_line.point1, border_line.point2, [border_line.point1], 0, False))
            recursion.append(border_line.point2)
            for point in recursion:
                indexes.append(Points.index(point))
        Faces.append(indexes)


def recursion_maze(start_point: list, finish_point: list, passed_points: list[list], i: int, can_use_border: bool):
    i = i + 1
    if i > 500:
        return False
    global Points
    if advanced_connected(start_point, finish_point, can_use_border) and start_point not in passed_points:
        passed_points.append(start_point)
        return passed_points
    connect_pointss = []
    close_point = []
    for point in Points:
        if point == start_point or point == finish_point:
            continue
        if point in passed_points:
            continue
        if advanced_connected(point, start_point, can_use_border):
            connect_pointss.append(point)
    for point in connect_pointss:
        if not close_point:
            close_point = point
            continue
        if distance(point, finish_point) < distance(close_point, finish_point):
            close_point = point
    if start_point not in passed_points:
        passed_points.append(start_point)
    return recursion_maze(close_point, finish_point, passed_points, i, can_use_border)


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
        if connect_line.point1 == point1 or connect_line.point1 == point2:
            if connect_line.point2 == point1 or connect_line.point2 == point2:
                return True
    return False


def fix_points():
    global Points, Lines, HEIGHT, ANGLE
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

        roof_point[2] = math.tan(ANGLE) * distance(roof_point, final_point) + final_point[2]


def distance(point1: list, point2: list):
    return math.sqrt(math.pow((point1[0] - point2[0]), 2) + math.pow(point1[1] - point2[1], 2))


def rebuild_points():
    global worthno_points, Points
    for point in Points:
        worthno_points.append(point)
    Points = worthno_points


def fine_make_a_file():
    global Points, Faces
    file_string = """CubePoints = [\n"""
    for point in Points:
        file_string = file_string + str(point)
        if Points.index(point) != len(Points) - 1:
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

    if PRINTING:
        print(file_string)


if __name__ == '__main__':
    if f.find("roof_height") == 0:
        Points = list(find_points(f))
        Faces = list(find_faces(f))
        Faces = list(not_top_faces(Points, Faces))
    else:
        Points = list(find_points(f))
        Faces = list(find_faces(f))
        HEIGHT = find_height(f)
        create_faces(Points)
        if HEIGHT != 0:
            top_points = list(add_top_points(Points, HEIGHT))
            for top_point in top_points:
                Points.append(top_point)

    if f.find("roof_angle") > 0:
        ANGLE = find_roof_angle(f)
    worthno_points = worthless_points(Points)
    Lines = list(border_lines(Points))
    # This is where the actual math begins
    draw(Points)

    setup_angles(Points)

    find_lines()
    unfinished_lines = find_unfinished_lines()
    connect_roof_points()
    '''
    try:
        connect_roof_points()
    except:
        print("ryú")'''
    refresh_encloseure()
    '''
    for point in Points:
        print(point)'''
    for line in Lines:
        print(line)
        draw_line(line, 'y')

    if DRAWING:
        plt.show()

    fix_points()
    rebuild_points()
    new_faces()

    fine_make_a_file()
