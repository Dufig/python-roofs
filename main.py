# import openpyscad as ops
import numpy as np  # v 1.19.2
import matplotlib.pyplot as plt  # v 3.3.2
import math

height = 0
f = open("better_kvadr.scad", "r")
f = f.read()
drawing = False
roof_points = []
rounding_error = 0.000000000000005


class Line:
    def __init__(self, point1: list, point2: list, beginning: list, border: bool):
        self.point1 = point1
        self.point2 = point2
        self.beginning = beginning
        self.border = border

    def slope(self):
        try:
            slope = (self.point2[1] - self.point1[1]) / (self.point2[0] - self.point1[0])
            return slope
        except ZeroDivisionError:
            return (self.point2[1] - self.point1[1]) / (abs(self.point2[1] - self.point1[1])) * 922336854775807

    def __eq__(self, other):
        if self.point1 == other.point1 and self.point2 == other.point2 and self.beginning == \
                other.beginning and self.border == other.border:
            return True
        return False

    def __str__(self):
        return str(self.point1) + str(self.point2) + str(self.beginning) + str(self.border)


def draw(points: list):
    # Enter x and y coordinates of points and colors
    xs = []
    ys = []
    colors = ['m']
    for point in points:
        xs.append(point[0])
        ys.append(point[1])

    # Select length of axes and the space between tick labels
    xmin, xmax, ymin, ymax = -15, 15, -15, 15
    ticks_frequency = 1

    # Plot points
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.scatter(xs, ys, c=colors)

    # Draw lines connecting points to axes
    for x, y, c in zip(xs, ys, colors):
        ax.plot([x, x], [0, y], c=c, ls='--', lw=1.5, alpha=0.5)
        ax.plot([0, x], [y, y], c=c, ls='--', lw=1.5, alpha=0.5)

    for point in points:
        if point == points[0]:
            ax.plot([point[0], points[points.index(point) + 1][0]], [point[1], points[points.index(point) + 1][1]],
                    c='m')
        elif point == points[len(points) - 1]:
            ax.plot([point[0], points[0][0]], [point[1], points[0][1]], c='m')
        else:
            ax.plot([point[0], points[points.index(point) + 1][0]], [point[1], points[points.index(point) + 1][1]],
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


def find_faces(no_list: str) -> list:
    faces = []
    no_list = no_list[no_list.index("Faces"):]
    no_list = no_list[no_list.index("[") + 1:no_list.index(";") - 1]
    no_list = no_list + ",L"
    coor = []
    number = int(no_list.count(",") / no_list.count("["))
    for i in range(no_list.count("[")):
        for j in range(number):
            temp = no_list[:no_list.index(",")]
            no_list = no_list[no_list.index(",") + 1:]
            if j == 0:
                temp = temp[temp.find("[") + 1:]
            if j == number - 1:
                temp = temp[:-1]
            temp = int(temp)
            coor.append(temp)
            if j == number - 1:
                faces.append(coor)
                coor = []

    return faces


def not_top_faces(points: list, faces: list) -> list:
    global height
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

    height = heights[len(heights) - 1] + heights[len(heights) - 1] - heights[0]
    return good_faces


def worthless_points(points: list) -> list:
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
    and adding them up
    """
    xa = a[0] - mid[0]
    ya = a[1] - mid[1]
    xmid = mid[0]
    ymid = mid[1]
    xc = c[0] - mid[0]
    yc = c[1] - mid[1]
    len_a = math.sqrt(math.pow(xa, 2) + math.pow(ya, 2))
    len_c = math.sqrt(math.pow(xc, 2) + math.pow(yc, 2))
    ratio = len_a / len_c
    if ratio != 1:
        xc = xc * ratio
        yc = yc * ratio
    Lines.append(Line([xc + xa + xmid, ya + yc + ymid, height], [- xc - xa + xmid, - ya - yc + ymid, height], mid,
                      False))
    return Line([xc + xa + xmid, ya + yc + ymid, height], [- xc - xa + xmid, - ya - yc + ymid, height], mid, False)


def border_lines(points: list) -> list:
    lines = []
    for point in points:
        if points.index(point) == len(points) - 1:
            lines.append(Line(point, Points[0], point, True))
        else:
            lines.append(Line(point, Points[Points.index(point) + 1], point, True))

    return lines


def setup_angles(points: list):
    for point in points:
        if points.index(point) == 0:
            find_angle(points[len(points) - 1], points[0], points[1])
        elif points.index(point) == len(points) - 1:
            find_angle(points[len(points) - 2], points[len(points) - 1], points[0])
        else:
            find_angle(points[points.index(point) - 1], points[points.index(point)], points[points.index(point) + 1])


def find_lines(lines):
    """This function finds the intercept of angle axis and proceeds to draw them. It goes through all the points,

    finds the ones next to the point, then angle lines beginning in the point and the intersection that is the closest.

    You would NOT believe how long it took me to fix all the bugs in this, as a result it is very cluttered.

    And yes, I know you are only supposed to go like 4 layers in before making more functions, leave me alone ok?
    """
    global Points, Lines, roof_points
    temp_points = []
    temp_lines = []
    true_point = []
    for point in Points:
        if Points.index(point) == 0:
            point_lines = [Points[len(Points) - 1], Points[1]]
        elif Points.index(point) == len(Points) - 1:
            point_lines = [Points[Points.index(point) - 1], Points[0]]
        else:
            point_lines = [Points[Points.index(point) - 1], Points[Points.index(point) + 1]]

        for angle_line in lines:
            if point == angle_line.beginning and not angle_line.border:
                len_inter = 999999999999
                for interline in Lines:
                    if not interline.border and (interline.beginning == point_lines[0]
                                                 or interline.beginning == point_lines[1]):
                        intersect = list(line_intersecting(angle_line, interline))

                        intersect.append(height)
                        if math.sqrt(math.pow((intersect[0] - angle_line.beginning[0]), 2)
                                     + math.pow(intersect[1] - angle_line.beginning[1], 2)) < len_inter:
                            len_inter = math.sqrt(math.pow((intersect[0] - angle_line.beginning[0]), 2)
                                                  + math.pow(intersect[1] - angle_line.beginning[1], 2))
                            true_point = intersect

                if true_point:
                    new_line = Line(angle_line.beginning, true_point, angle_line.beginning, False)
                    temp_lines.append(new_line)
                    draw_line(new_line, 'r')

    for point in temp_points:
        Points.append(point)
        roof_points.append(point)
    deround_lines(temp_lines)
    for drawn_line in temp_lines:
        if temp_points.count(drawn_line.point2) == 0:
            temp_points.append(drawn_line.point2)
            roof_points.append(drawn_line.point2)
            plt.scatter(drawn_line.point2[0], drawn_line.point2[1], c='r')
    for axis_line in Lines:
        for true_line in temp_lines:
            if axis_line.beginning == true_line.point1 and not axis_line.border:
                axis_line.beginning = true_line.beginning
                axis_line.point1 = true_line.point1
                axis_line.point2 = true_line.point2


def deround_lines(lines_list: list[Line]):
    global rounding_error
    for first_line in lines_list:
        list1 = [first_line.point1, first_line.point2]
        for second_line in lines_list:
            list2 = [second_line.point1, second_line.point2]
            for point1 in list1:
                for point2 in list2:
                    if abs(point1[0] - point2[0]) < rounding_error and abs(point1[1] - point2[1]) < rounding_error:
                        if first_line.point1 == point1:
                            first_line.point1 = point2
                        elif first_line.point1 == point2:
                            first_line.point2 = point2
                        elif second_line.point1 == point2:
                            second_line.point1 = point1
                        elif second_line.point2 == point2:
                            second_line.point2 = point1


def deround(bad_points: list):
    global rounding_error
    for point in bad_points:
        if bad_points.count(point) > 1:
            bad_points.remove(point)
    for first_point in bad_points:
        for second_point in bad_points:
            if first_point == second_point:
                pass
            elif abs(first_point[0] - second_point[0]) < rounding_error and abs(first_point[1] - second_point[1]) < \
               rounding_error:
                bad_points.remove(first_point)


def draw_line(aline: Line, color):
    plt.plot([aline.point1[0], aline.point2[0]], [aline.point1[1], aline.point2[1]], c=color, ls='-', lw=1.5, alpha=0.5)


def connect_roof_points():
    global roof_points, Points, Lines
    if len(roof_points) == 2:
        Lines.append(Line(roof_points[0], roof_points[1], roof_points[0], False))
        draw_line(Lines[len(Lines) - 1], 'k')
        Points.append(roof_points[0])
        Points.append(roof_points[1])
        return None
    bordering_lines = []
    for point in roof_points:
        bordering_points = []
        for connecting_line in Lines:
            if connecting_line.point2 == point and connecting_line.point1 not in roof_points:
                bordering_points.append(connecting_line.point1)
        for border_point in bordering_points:
            for border_line in Lines:
                if (border_line.point1 == border_point or border_line.point2 == border_point) and border_line.border:
                    if border_line not in bordering_lines:
                        bordering_lines.append(border_line)
                    else:
                        bordering_lines.remove(border_line)
        if len(bordering_lines) == 2:
            line1 = bordering_lines[0]
            line2 = bordering_lines[1]
            roof_line = find_angle(line1.point1, line_intersecting(line1, line2), line2.point2)
            roof_line.beginning = point
            roof_line = shorten_line(roof_line, point)

            draw_line(roof_line, 'r')
        for line in bordering_lines:
            print(line.point1, line.point2)


def shorten_line(long_line: Line, point: list):
    if (math.sqrt(math.pow((point[0] - long_line.point1[0]), 2) + math.pow(point[1] - long_line.point1[1], 2)) <
       math.sqrt(math.pow((point[0] - long_line.point1[0]), 2) + math.pow(point[1] - long_line.point1[1], 2))):
        long_line.point2 = long_line.beginning
    else:
        long_line.point1 = long_line.beginning
    return long_line


def y_intercept(p1, slope):
    return p1[1] - slope * p1[0]


def line_intersecting(line1: Line, line2: Line):
    m1 = line1.slope()
    m2 = line2.slope()
    if m1 == m2:
        return None
    b1 = y_intercept(line1.point1, m1)
    b2 = y_intercept(line2.point1, m2)
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return [x, y]


def new_faces():
    global Lines, Points, Faces
    true_lines = []
    for line in Lines:
        if line.border:
            true_lines.append(line)
    for border_line in true_lines:
        face_points = [border_line.point1]
        indexes = []
        recursion = list(recursion_maze(border_line.point2, border_line.point1, [border_line.point2]))
        for point in recursion:
            face_points.append(point)
        for point in face_points:
            indexes.append(Points.index(point))
        Faces.append(indexes)


def recursion_maze(start_point: list, finish_point: list, passed_points: list[list]):
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
    return recursion_maze(close_point, finish_point, passed_points)




def connected(point1: list, point2: list):
    global Lines
    for line in Lines:
        if line.point1 == point1 or line.point1 == point2:
            if line. point2 == point1 or line.point2 == point2:
                return True
    return False


def distance(point1: list, point2: list):
    return math.sqrt(math.pow((point1[0] - point2[0]), 2) + math.pow(point1[1] - point2[1], 2))


def rebuild_points():
    global worthno_points, Points
    for point in Points:
        worthno_points.append(point)
    Points = worthno_points

if __name__ == '__main__':
    Points = list(find_points(f))
    Faces = list(find_faces(f))
    Faces = list(not_top_faces(Points, Faces))
    worthno_points = worthless_points(Points)
    Lines = list(border_lines(Points))
    '''This all above is just basic file formatting, getting coordinates from the file and things as such, nothing 
    too important but it does take up a lot of space as I couldn't find a python library that would fit my needs'''
    draw(Points)
    setup_angles(Points)

    find_lines(Lines)
#    for line in Lines:
#        print(line.point1, line.point2, line.beginning, line.border)
    connect_roof_points()

    rebuild_points()
    new_faces()
    print(Points)
    print(Faces)
    if drawing:
        plt.show()
