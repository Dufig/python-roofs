import numpy as np  # Only used for graphing
import matplotlib.pyplot as plt  # Only used for graphing
import math

height = 0
f = open("6stran_podstava.scad", "r")
f = f.read()
drawing = True
printing = True
roof_points = []
used_lines = []
rounding_error = 0.00000000000005
angle = math.radians(45)


class Line:
    def __init__(self, point1: list, point2: list, beginning: list, border: bool, final: bool):
        self.point1 = point1
        self.point2 = point2
        self.beginning = beginning
        self.border = border
        self.final = final

    def slope(self):

        if self.point1[0] < self.point2[0]:
            try:
                slope = (self.point2[1] - self.point1[1]) / (self.point2[0] - self.point1[0])
                return slope
            except ZeroDivisionError:
                return abs((self.point2[1] - self.point1[1]) / (abs(self.point2[1] - self.point1[1]))) * 999999999999999
        else:
            try:
                slope = (self.point1[1] - self.point2[1]) / (self.point1[0] - self.point2[0])
                return slope
            except ZeroDivisionError:
                return abs((self.point1[1] - self.point2[1]) / (abs(self.point1[1] - self.point2[1]))) * 999999999999999

    def __eq__(self, other):
        if self.point1 == other.point1 and self.point2 == other.point2 and self.beginning == \
                other.beginning and self.border == other.border:
            return True
        return False

    def __str__(self):
        return str(self.point1) + str(self.point2) + str(self.beginning) + str(self.border) + str(self.final)

    def lenght(self):
        return math.sqrt(math.pow((self.point1[0] - self.point2[0]), 2) + math.pow(self.point1[1] - self.point2[1], 2))


def draw(points: list):
    # Enter x and y coordinates of points and colors
    xs = []
    ys = []
    colors = ['m']
    for top_point in points:
        xs.append(top_point[0])
        ys.append(top_point[1])

    # Select length of axes and the space between tick labels
    xmin, xmax, ymin, ymax = -15, 15, -15, 15
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
            temp = int(temp)
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
    if no_list.count("height_coefficient") != 1:
        return None
    no_list = no_list[no_list.index("height_coefficient"):]
    no_list = no_list[no_list.index("=") + 1:]
    no_list = no_list[:no_list.index(";")]
    no_list = float(no_list)
    return no_list


def not_top_faces(points: list, faces: list) -> list:
    # Gets rid of the top face, as it will get covered by the roof
    global height, f
    heights = []
    bad_points = []
    good_faces = []
    for point in points:
        if point[2] not in heights:
            heights.append(point[2])
    top = max(heights)
    for i in range(len(points)):
        if not (points[i][2] >= top):
            bad_points.append(points[i])
    for face in faces:
        for point in face:
            if points[point] in bad_points:
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
                [xmid, ymid, zmid], False,
                False)


'''
Line([xc + xa + xmid, ya + yc + ymid, height], [- xc - xa + xmid, - ya - yc + ymid, height],
                [xmid, ymid, height], False, False)'''


def border_lines(points: list) -> list:
    lines = []
    for point in points:
        if points.index(point) == len(points) - 1:
            lines.append(Line(point, Points[0], point, True, False))
        else:
            lines.append(Line(point, Points[Points.index(point) + 1], point, True, False))

    return lines


def setup_angles(points: list):
    # Just a setup for the find_angle function.
    for point in points:
        if points.index(point) == 0:
            line = find_angle(points[len(points) - 1], points[0], points[1])
        elif points.index(point) == len(points) - 1:
            line = find_angle(points[len(points) - 2], points[len(points) - 1], points[0])
        else:
            line = find_angle(points[points.index(point) - 1], points[points.index(point)],
                              points[points.index(point) + 1])
        Lines.append(line)


def find_lines():
    """This function finds the intercept of angle axis and proceeds to draw them. It goes through all the points,

    finds the ones next to the point, then angle lines beginning in the point and the intersection that is the closest.

    You would NOT believe how long it took me to fix all the bugs in this, as a result it is very cluttered.

    Alright, future me here, turned out the function here was super messy and only worked for very basic stuff, so I

    re-wrote the entire thing. Works still pretty much the same except it checks if the intersection is the closest

    intersection for *both* lines now, which was necessary for more complex houses.
    """
    global Points, Lines, roof_points

    temp_points = []
    temp_lines = []
    true_lines = []
    for point in Points:
        possible_lines = []
        for a_line in Lines:
            if a_line.beginning == point and not a_line.border:
                a_line = a_line
                if Points.index(point) == 0:
                    point_lines = [Points[len(Points) - 1], Points[1]]
                elif Points.index(point) == len(Points) - 1:
                    point_lines = [Points[Points.index(point) - 1], Points[0]]
                else:
                    point_lines = [Points[Points.index(point) - 1], Points[Points.index(point) + 1]]
                for line_point in point_lines:
                    for interline in Lines:
                        if interline.beginning == line_point and not interline.border and not interline.final:
                            possible_lines.append(interline)
                for possible_line in possible_lines:
                    if not line_intersecting(possible_line, a_line):
                        possible_lines.remove(possible_line)
                for possible_line in possible_lines:
                    if not line_intersecting(possible_line, a_line):
                        possible_lines.remove(possible_line)
                for possible_line in possible_lines:
                    if not are_closest(a_line, possible_line):
                        possible_lines.remove(possible_line)
                for possible_line in possible_lines:
                    if len(possible_lines) == 1:
                        pair = [possible_line, a_line]
                        temp_lines.append(pair)
                    else:
                        if len(possible_lines) == 2:
                            if distance(line_intersecting(possible_lines[0], a_line), a_line.beginning) < \
                                    distance(line_intersecting(possible_lines[1], a_line), a_line.beginning):
                                pair = [possible_lines[0], a_line]
                            else:
                                pair = [possible_lines[1], a_line]
                            temp_lines.append(pair)

    for first_pair in temp_lines:
        for second_pair in temp_lines:
            if first_pair == second_pair[::-1]:
                first_line = first_pair[0]
                second_line = second_pair[0]
                intersect = list(line_intersecting(first_line, second_line))
                intersect[2] = math.tan(angle) * distance(intersect, first_line.beginning) + first_line.beginning[2]
                first_line.point1 = first_line.beginning
                first_line.point2 = intersect
                first_line.final = True
                second_line.point1 = second_line.beginning
                second_line.point2 = intersect
                second_line.final = True
                true_lines.append(first_line)
                true_lines.append(second_line)
                temp_points.append(intersect)
                plt.scatter(intersect[0], intersect[1], c='r')

    for drawn_line in true_lines:
        if Points.count(drawn_line.point2) == 0:
            roof_points.append(drawn_line.point2)
            Points.append(drawn_line.point2)
            plt.scatter(drawn_line.point2[0], drawn_line.point2[1], c='r')
    for axis_line in Lines:
        for true_line in true_lines:
            if axis_line.beginning == true_line.point1 and not axis_line.border:
                axis_line.beginning = true_line.beginning
                axis_line.point1 = true_line.point1
                axis_line.point2 = true_line.point2
                draw_line(axis_line, 'r')


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
                return distance(line_intersecting(second_line, cool_line), cool_line.beginning) \
                    > distance(line_intersecting(angle_line, cool_line), cool_line.beginning)


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
    """
     Finds the points that were made by intersecting lines and that don't have a line connecting them to other points

    which are also a part of the top of the roof. Then it finds the correct line and makes it a continuation, finds the

    first time it intersects a line that isn't finished yet. Rinse and repeat and you have a finished roof. There is

    a shortcut at the beginning for when there are only two points to connect.
    """
    global roof_points, Points, Lines, height, unfinished_lines, used_lines, angle
    refresh_encloseure()
    if len(roof_points) == 2 and len(unfinished_lines) == 0:
        Lines.append(Line(roof_points[0], roof_points[1], roof_points[0], False, True))
        draw_line(Line(roof_points[0], roof_points[1], roof_points[0], False, True), 'r')
        # draw_line(Lines[len(Lines) - 1], 'r')
        if roof_points[0] not in Points:
            Points.append(roof_points[0])
        if roof_points[1] not in Points:
            Points.append(roof_points[1])
        return None
    roof_points = clear_doubles(roof_points)
    delete_points = []
    add_points = []
    for point in roof_points:
        bordering_lines = []
        bordering_points = []
        for border_point in Points:
            if connected(border_point, point):
                bordering_points.append(border_point)

        for border_point in bordering_points:
            for border_line in Lines:
                if (border_line.point1 == border_point or border_line.point2 == border_point) and \
                        border_line.border and not border_line.final:
                    if border_line not in bordering_lines:
                        bordering_lines.append(border_line)
        if len(bordering_lines) == 2:
            if not abs(Lines.index(bordering_lines[0]) - Lines.index(bordering_lines[1])) == 1:
                line1 = bordering_lines[0]
                line2 = bordering_lines[1]
                if line1.slope() != line2.slope():
                    roof_line = find_angle(line1.point1, line_intersecting(line1, line2), line2.point2)
                    if distance(roof_line.point2, point) > distance(roof_line.point1, point):
                        roof_line.point1 = point
                        roof_line.beginning = point
                    else:
                        roof_line.point2 = point
                        roof_line.beginning = point
                else:
                    if connected(line1.point1, point):
                        roof_line = Line(point,
                                         [point[0] + line1.point2[0] - line1.point1[0], point[1] + line1.point2[1] -
                                          line1.point1[1], height], point, False, False)
                    elif connected(line1.point2, point):
                        roof_line = Line(point,
                                         [point[0] + line1.point1[0] - line1.point2[0], point[1] + line1.point1[1] -
                                          line1.point2[1], height], point, False, False)
            else:
                check_points = []

                for check_line in bordering_lines:
                    if check_line.point1 not in check_points:
                        check_points.append(check_line.point1)
                    else:
                        check_points.remove(check_line.point1)
                        corner_point = check_line.point1
                    if check_line.point2 not in check_points:
                        check_points.append(check_line.point2)
                    else:
                        check_points.remove(check_line.point2)
                        corner_point = check_line.point2
                for connecting_line in Lines:
                    if not connecting_line.border and connecting_line.final and (connecting_line.point1 == point
                                                                                 or connecting_line.point2 == point):
                        if not (connecting_line.point1 == corner_point or connecting_line.point2 == corner_point):
                            if connecting_line.point1 == point:
                                bad_point = connecting_line.point2
                            elif connecting_line.point2 == point:
                                bad_point = connecting_line.point1
                len_check = 0  # this could lead to a problem, if there is a bug check this part
                for check_point in check_points:  # its good as a placeholder though
                    if distance(check_point, bad_point) > len_check:
                        len_check = distance(check_point, bad_point)
                        good_point = check_point
                for intersecting_line in Lines:
                    if intersecting_line.border and (
                            intersecting_line.point1 == good_point or intersecting_line.point2 == good_point) and (
                            intersecting_line.point1 == corner_point or intersecting_line.point2 == corner_point):
                        intersect_line = intersecting_line
                        break
                for second_line in Lines:
                    if second_line.border and line_intersecting(intersect_line, second_line) and \
                            not second_line == intersect_line and second_line.point1 != corner_point and \
                            second_line.point2 != corner_point and not second_line.final:
                        if corner_point[0] <= good_point[0]:
                            if corner_point[1] <= good_point[1]:
                                if line_intersecting(intersect_line, second_line)[0] <= corner_point[
                                    0] + rounding_error and \
                                        line_intersecting(intersect_line, second_line)[1] <= corner_point[
                                         1] + rounding_error:
                                    bor_itr_line = second_line
                            if corner_point[1] >= good_point[1]:
                                if line_intersecting(intersect_line, second_line)[0] <= corner_point[
                                    0] + rounding_error and \
                                        line_intersecting(intersect_line, second_line)[1] + rounding_error >= \
                                        corner_point[1]:
                                    bor_itr_line = second_line
                        elif corner_point[1] < good_point[1]:
                            if corner_point[0] <= good_point[0]:
                                if line_intersecting(intersect_line, second_line)[1] < corner_point[
                                    1] + rounding_error and \
                                        line_intersecting(intersect_line, second_line)[0] <= corner_point[
                                          0] + rounding_error:
                                    bor_itr_line = second_line
                            if corner_point[0] >= good_point[0]:
                                if line_intersecting(intersect_line, second_line)[1] < corner_point[
                                    1] + rounding_error and \
                                        line_intersecting(intersect_line, second_line)[0] + rounding_error >= \
                                        corner_point[0]:
                                    bor_itr_line = second_line

                if distance(bor_itr_line.point1, bad_point) < distance(bor_itr_line.point2, bad_point):
                    roof_line = find_angle(bor_itr_line.point2, line_intersecting(intersect_line, bor_itr_line),
                                           corner_point)
                else:
                    roof_line = find_angle(bor_itr_line.point1, line_intersecting(intersect_line, bor_itr_line),
                                           corner_point)
                if distance(point, roof_line.point1) < distance(point, roof_line.point2):
                    roof_line.beginning = point
                    roof_line.point2 = point
                else:
                    roof_line.beginning = point
                    roof_line.point1 = point
        len_a = 9999999999
        for passing_line in Lines:
            if line_intersecting(passing_line, roof_line) and not passing_line.final and not passing_line.border:
                if distance(roof_line.beginning, line_intersecting(passing_line, roof_line)) < len_a:
                    interline = passing_line
                    len_a = distance(roof_line.beginning, line_intersecting(passing_line, roof_line))
        intersect = line_intersecting(interline, roof_line)
        intersect[2] = math.tan(angle) * distance(intersect, interline.beginning) + interline.beginning[2]
        interline.point1 = interline.beginning
        interline.point2 = intersect
        interline.final = True
        roof_line.point2 = intersect
        roof_line.point1 = roof_line.beginning
        roof_line.final = True
        roof_points.remove(roof_line.beginning)
        if roof_line.beginning not in Points:
            Points.append(roof_line.beginning)
            """
            There was a bug involving a point getting into roof points but the going missing while never getting into 
            Points, this is there as a failsafe. The point was a correct intersect though.
            """
        add_points.append(intersect)
        draw_line(roof_line, 'r')
        draw_line(interline, 'r')
        plt.scatter(intersect[0], intersect[1], c='r')
        Lines.append(roof_line)
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
    if len(unfinished_lines) != 0 or len(roof_points) != 0:
        connect_roof_points()


def refresh_encloseure():
    global Lines
    for refresh_line in Lines:
        if refresh_line.border and not refresh_line.final:
            refresh_line.final = check_enclosed(refresh_line)


def check_enclosed(border_line: Line) -> bool:
    global Lines
    close_points = recursion_maze(border_line.point1, border_line.point2, [border_line.point1], 0)
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
    return [x, y, height]


def new_faces():
    global Lines, Points, Faces
    true_lines = []
    for line in Lines:
        if line.border:
            true_lines.append(line)
    for border_line in true_lines:
        face_points = [border_line.point1]
        indexes = []
        recursion = list(recursion_maze(border_line.point2, border_line.point1, [border_line.point2], 0))
        for point in recursion:
            face_points.append(point)
        for point in face_points:
            indexes.append(Points.index(point))
        Faces.append(indexes)


def recursion_maze(start_point: list, finish_point: list, passed_points: list[list], i: int):
    i = i + 1
    if i > 500:
        return False
    global Points
    if connected(start_point, finish_point) and start_point not in passed_points:
        passed_points.append(start_point)
        return passed_points
    connect_pointss = []
    close_point = []
    for point in Points:
        if point == start_point or point == finish_point:
            continue
        if point in passed_points:
            continue
        if connected(point, start_point):
            connect_pointss.append(point)

    for point in connect_pointss:
        if not close_point:
            close_point = point
            continue
        if distance(point, finish_point) < distance(close_point, finish_point):
            close_point = point
    if start_point not in passed_points:
        passed_points.append(start_point)
    return recursion_maze(close_point, finish_point, passed_points, i)


def connected(point1: list, point2: list):
    global Lines
    for line in Lines:
        if line.point1 == point1 or line.point1 == point2:
            if line.point2 == point1 or line.point2 == point2:
                return True
    return False


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
    Points = list(find_points(f))
    Faces = list(find_faces(f))
    find_height(f)
    Faces = list(not_top_faces(Points, Faces))
    worthno_points = worthless_points(Points)
    Lines = list(border_lines(Points))

    # This is where the actual math begins
    draw(Points)
    setup_angles(Points)

    find_lines()
    unfinished_lines = find_unfinished_lines()
    connect_roof_points()
    #fix_height()
    # for point in Points:
    # print(point)
    #for line in Lines:
        #print(line)
        #draw_line(line, 'y')
    rebuild_points()
    new_faces()

    fine_make_a_file()

    if drawing:
        plt.show()


