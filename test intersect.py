import numpy as np  # Only used for graphing
import matplotlib.pyplot as plt  # Only used for graphing
import math

height = 0
f = open("Big_flat_house.scad", "r")
f = f.read()
drawing = False
printing = True
roof_points = []
used_lines = []
rounding_error = 0.00000000005
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
    ticks_frequency = 1

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
    for point in points:
        if point[2] not in heights:
            heights.append(point[2])
    top = max(heights)
    for i in range(len(points)):
        if not (points[i][2] >= top):
            bad_points.append(points[i])
    for point in bad_points:
        points.remove(point)
    return bad_points


def add_top_points(points: list[list], tallness: int or float) -> list:
    upper_points = []
    for point in points:
        new_point = [point[0], point[1], point[2] + tallness]
        upper_points.append(new_point)
    return upper_points


def create_faces(low_points: list[list]):
    global Faces
    face = []
    if len(Faces) != 1:
        for point in low_points:
            face.append(low_points.index(point))
        Faces = [face]
    for point in low_points:
        if low_points.index(point) == len(low_points) - 1:
            face = [low_points.index(point), 2 * len(low_points) - 1, len(low_points), 0]
        else:
            face = [low_points.index(point), len(low_points) + low_points.index(point), len(low_points) +
                    low_points.index(point) + 1, low_points.index(point) + 1]
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
    lines = []
    for point in points:
        walls = [Points.index(point) + 1, Points.index(point) + 1]
        if points.index(point) == len(points) - 1:
            lines.append(Line(point, Points[0], point, True, False, walls))
        else:
            lines.append(Line(point, Points[Points.index(point) + 1], point, True, False, walls))

    return lines


def setup_angles(points: list):
    # Just a setup for the find_angle function.
    for point in points:
        wall_lines = []
        if points.index(point) == 0:
            angle_line = find_angle(points[len(points) - 1], points[0], points[1])
        elif points.index(point) == len(points) - 1:
            angle_line = find_angle(points[len(points) - 2], points[len(points) - 1], points[0])
        else:
            angle_line = find_angle(points[points.index(point) - 1], points[points.index(point)],
                                    points[points.index(point) + 1])
        for wall_line in Lines:
            if wall_line.border and (wall_line.point1 == point or wall_line.point2 == point):
                wall_lines.append(wall_line)
        if len(wall_lines) == 2:
            walls = get_walls(wall_lines[0], wall_lines[1])
            angle_line.walls = walls
        Lines.append(angle_line)


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
        if not line_intersecting(temp_line, border_line):
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
    amount_changed = 0

    axis_lines = find_intersects(axis_lines)



def find_intersects(axis_lines: list[Line]):
    global Points, Lines, rounding_error, roof_points
    intersected_lines = []
    for axis_line in axis_lines:
        intesects = [axis_line]
        for second_line in axis_lines:
            if axis_line == second_line:
                continue
            if axis_line.final:
                continue
            if type(line_intersecting(second_line, axis_line)) == bool and line_intersecting(second_line, axis_line):
                intesects.append(second_line)
                continue
            if line_intersecting(second_line, axis_line):
                print("lllllll", line_intersecting(second_line, axis_line))
                if check_crossing(axis_line.beginning, line_intersecting(second_line, axis_line)):
                    continue
                if check_crossing(second_line.beginning, line_intersecting(second_line, axis_line)):
                    continue
                if not check_in_shape(line_intersecting(second_line, axis_line)):
                    continue
                intesects.append(second_line)
        intersected_lines.append(intesects)
    for list in intersected_lines:
        print(len(list))
    print(" ")
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
            axis_list = [axis_line]
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
        print("remove", len(remove_lines))
        for second_line in remove_lines:
            axis_list.remove(second_line)
    for list in intersected_lines:
        print(len(list), list[0])

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


    intesects = []
    for inter_list in intersected_lines:
        intersect = []
        axis_line = inter_list[0]
        for second_line in inter_list:
            if second_line == axis_line:
                continue
            if type(line_intersecting(second_line, axis_line)) == bool:
                continue
            intersect = line_intersecting(second_line, axis_line)
        if intersect != []:
            intersect[2] = height
            if abs(round(intersect[0]) - intersect[0]) < rounding_error:
                intersect[0] = float(round(intersect[0]))
            if abs(round(intersect[1]) - intersect[1]) < rounding_error:
                intersect[1] = float(round(intersect[1]))
            if intersect not in Points:
                Points.append(intersect)
            if intersect not in roof_points:
                roof_points.append(intersect)
            plt.scatter(intersect[0], intersect[1], c='r')
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
                draw_line(second_line, 'r')
                axis_lines[index] = second_line
            index = axis_lines.index(axis_line)
            axis_line.point1 = axis_line.beginning
            axis_line.point2 = intersect
            axis_line.final = True
            axis_lines[index] = axis_line
            draw_line(axis_line, 'r')
    return axis_lines


def are_closest(angle_line: Line, cool_line: Line) -> bool:
    """
    Checks if intersect of the angle line and the cool line are the closest intersect of cool line in relation to other

    lines that are not borderlines.
    """
    global Points, Lines
    if not line_intersecting(angle_line, cool_line):
        return False
    point = cool_line.beginning
    if Points.index(point) == 0:
        corner_points = [Points[1], Points[len(Points) - 1]]
    elif Points.index(point) == len(Points) - 1:
        corner_points = [Points[0], Points[len(Points) - 2]]
    else:
        corner_points = [Points[Points.index(point) - 1], Points[Points.index(point) + 1]]
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
    for point in points:
        while points.count(point) > 1:
            points.remove(point)

    return points


def connect_roof_points():
    global roof_points, Points, Lines, height, unfinished_lines, used_lines, angle
    print(roof_points, "roof")
    if len(roof_points) <= 2 and len(unfinished_lines) == 0:
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
    add_points = []
    delete_points = []
    for point in roof_points:
        refresh_encloseure()
        connecting_lines = []
        walls = []

        for connecting_line in Lines:
            if connecting_line.point1 == point or connecting_line.point2 == point:
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
            if border_line1.point1 != point:
                point1 = border_line1.point1
            else:
                point1 = border_line1.point2
            if border_line2.point1 != point:
                point2 = border_line2.point1
            else:
                point2 = border_line2.point2
            roof_line = find_angle(point1, line_intersecting(border_line1, border_line2), point2)
            roof_line.beginning = point
            roof_line.walls = walls
        else:
            roof_line = Line(point, [border_line1.point1[0] - border_line1.point2[0] + point[0], border_line1.point1[1]
                                     - border_line1.point2[1] + point[1], point[2]], point, False, False, walls)
        roof_lines.append(roof_line)

    roof_walls = []
    for roof_line in roof_lines:
        if roof_line.walls in roof_walls:
            roof_lines.remove(roof_line)
        roof_walls.append(roof_line.walls)
    for roof_line in roof_lines:
        len_a = 0
        interlines = []
        for second_line in Lines:
            if not line_intersecting(roof_line, second_line) or second_line == roof_line or second_line.final or \
                    second_line.border:
                continue
            len_a = distance(roof_line.beginning, line_intersecting(roof_line, second_line))
            break
        for second_line in Lines:
            if not line_intersecting(roof_line, second_line) or second_line == roof_line or second_line.final or \
                    second_line.border:
                continue
            if distance(roof_line.beginning, line_intersecting(roof_line, second_line)) < len_a + rounding_error:
                len_a = distance(roof_line.beginning, line_intersecting(roof_line, second_line))
        for second_line in Lines:
            if not line_intersecting(roof_line, second_line) or second_line == roof_line or second_line.border or \
                    second_line.final:
                continue
            if distance(roof_line.beginning, line_intersecting(roof_line, second_line)) <= len_a + rounding_error:
                interlines.append(second_line)
        if len(interlines) == 0:
            continue
        if len(interlines) == 1:
            interline = interlines[0]
            intersect = line_intersecting(roof_line, interline)
            if not check_in_shape(intersect):
                continue

            intersect[2] = height
            if abs(round(intersect[0]) - intersect[0]) < rounding_error:
                intersect[0] = float(round(intersect[0]))
            if abs(round(intersect[1]) - intersect[1]) < rounding_error:
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
                if abs(first_intersect[0] - intersect[0]) < rounding_error and abs(first_intersect[1] - intersect[1]) \
                        < rounding_error:
                    continue
                for interline in interlines:
                    if not line_intersecting(roof_line, interline):
                        continue
                    if line_intersecting(roof_line, interline) == intersect:
                        interlines.remove(interline)
            intersect = first_intersect
            if abs(round(intersect[0]) - intersect[0]) < rounding_error:
                intersect[0] = float(round(intersect[0]))
            if abs(round(intersect[1]) - intersect[1]) < rounding_error:
                intersect[1] = float(round(intersect[1]))
            intersect[2] = height
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
        '''

        for connecting_line in Lines:
            if connecting_line.point1 == point or connecting_line.point2 == point:
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
        for wall_line in Lines:
            if wall_line.walls[0] == wall_line.walls[1] and wall_line.border and \
                    (wall_line.walls[0] == walls[0] or wall_line.walls[0] == walls[1]):
                wall_lines.append(wall_line)
        #for line in connecting_lines:
        #print(line, "con")
        if line_intersecting(wall_lines[0], wall_lines[1]):
            roof_line = find_angle(wall_lines[0].point1, line_intersecting(wall_lines[0],
                                                                           wall_lines[1]), wall_lines[1].point2)

            roof_line.walls = walls
            if distance(point, roof_line.point1) < distance(point, roof_line.point2):
                roof_line.beginning = point
                roof_line.point2 = point
            else:
                roof_line.beginning = point
                roof_line.point1 = point
        else:
            line1 = wall_lines[0]
            line2 = wall_lines[1]
            if distance(line1.point1, line1.point2) > distance(line2.point1, line2.point2):
                long_line = line1
            else:
                long_line = line2
            if distance(long_line.point1, point) > distance(long_line.point2, point):
                roof_line = Line(point, [long_line.point1[0] - long_line.point2[0] + point[0], long_line.point1[1]
                                         - long_line.point2[1] + point[1], point[2]], point, False, False,
                                 [line1.walls[0], line2.walls[0]])
            else:
                roof_line = Line(point, [long_line.point2[0] - long_line.point1[0] + point[0], long_line.point2[1]
                                         - long_line.point1[1] + point[1], point[2]], point, False, False,
                                 [line1.walls[0], line2.walls[0]])




        len_a = 9999999999
        for passing_line in Lines:
            if line_intersecting(passing_line, roof_line) and not passing_line.final and not passing_line.border and passing_line.point1 != point and passing_line.point2 != point:
                if distance(roof_line.beginning, line_intersecting(passing_line, roof_line)) < len_a and \
                        check_in_shape(line_intersecting(passing_line, roof_line)):
                    interline = passing_line
                    len_a = distance(roof_line.beginning, line_intersecting(passing_line, roof_line))
        intersect = line_intersecting(interline, roof_line)
        if roof_line.beginning not in Points:
            Points.append(roof_line.beginning)
            """
              There was a bug involving a point getting into roof points but the going missing while never getting into 
              Points, this is there as a failsafe. The point was a correct intersect though.
              """
        if not check_in_shape(intersect):
            continue
        intersect[2] = height
        if abs(round(intersect[0]) - intersect[0]) < rounding_error:
            intersect[0] = float(round(intersect[0]))
        if abs(round(intersect[1]) - intersect[1]) < rounding_error:
            intersect[1] = float(round(intersect[1]))
        for dupli_point in Points:
            if dupli_point == intersect:
                Points.remove(dupli_point)
        interline.point1 = interline.beginning
        interline.point2 = intersect
        interline.final = True
        roof_line.point2 = intersect
        roof_line.point1 = roof_line.beginning
        roof_line.final = True
        add_points.append(intersect)
        roof_points.remove(roof_line.beginning)
        Lines.append(roof_line)
        print(roof_line, "ruff")
        print(interline)
        print(roof_points, intersect, point)
        roof_points.append(intersect)
        plt.scatter(intersect[0], intersect[1], c='r')

        unfinished_lines = find_unfinished_lines()
        if len(roof_points) <= 2 and len(unfinished_lines) == 0:
            connect_roof_points()
    for point in roof_points:
        for delete in delete_points:
            if point == delete:
                roof_points.remove(point)
                delete_points.remove(point)
    for point in add_points:
        if roof_points.count(point) < 1:
            roof_points.append(point)
            add_points.remove(point)
    unfinished_lines = find_unfinished_lines()
    refresh_encloseure()
    if len(unfinished_lines) != 0 or len(roof_points) != 0:
        connect_roof_points()'''


def check_in_shape(point: list) -> bool:
    crossed_lines = []
    point_line = Line(point, [point[0] + 10, point[1], point[2]], point, False, False, [0, 0])
    for crossed_line in Lines:
        if line_intersecting(point_line, crossed_line) and crossed_line.border:
            if type(line_intersecting(point_line, crossed_line)) == bool:
                crossed_lines.append(crossed_line)
                continue
            if line_intersecting(point_line, crossed_line)[0] < point[0]:
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
    print(close_points)
    return True


def shorten_line(long_line: Line, point: list):
    if (math.sqrt(math.pow((point[0] - long_line.point1[0]), 2) + math.pow(point[1] - long_line.point1[1], 2)) <
            math.sqrt(math.pow((point[0] - long_line.point1[0]), 2) + math.pow(point[1] - long_line.point1[1], 2))):
        long_line.point2 = long_line.beginning
    else:
        long_line.point1 = long_line.beginning
    return long_line


def line_intersecting(line1: Line, line2: Line):
    global rounding_error, height
    a1 = line1.slope()
    a2 = line2.slope()

    if a1 == 999999999999999:
        if a2 == 999999999999999:
            if abs(line1.point1[0] - line2.point1[0]) <= 2 * rounding_error:
                return True
            return False

        b2 = line2.point1[1] - (line2.slope() * line2.point1[0])
        x = line1.point1[0]
        y = a2 * x + b2

    elif a2 == 999999999999999:
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
    if printing:
        print(file_string)


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
    print(worthno_points)
    # This is where the actual math begins
    draw(Points)

    setup_angles(Points)

    find_lines()

    print(" ")
    print("heya")
    for line in Lines:
        for second_line in Lines:
            if line == second_line:
                continue
            if line_intersecting(line, second_line) == line_intersecting(second_line, line):
                continue
            if line_intersecting(line, second_line) and line_intersecting(second_line, line):
                if round(line_intersecting(line, second_line)[0]) == round(line_intersecting(second_line, line)[0]) and\
                round(line_intersecting(line, second_line)[1]) == round(line_intersecting(second_line, line)[1]):
                    continue
            print(line_intersecting(line, second_line), line, second_line)
            print(line_intersecting(second_line, line))
    print(" ")

    unfinished_lines = find_unfinished_lines()
    '''
    connect_roof_points()
    '''
    try:
        connect_roof_points()
    except:
        plt.show()

    refresh_encloseure()

    for point in Points:
        print(point)
    for line in Lines:
        print(line)

    if drawing:
        plt.show()

    fix_points()
    rebuild_points()
    new_faces()

    fine_make_a_file()