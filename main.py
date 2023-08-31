import numpy
import openpyscad as ops
import numpy as np                 # v 1.19.2
import matplotlib.pyplot as plt    # v 3.3.2
import math

f = open("kvadr.scad", "r")
f = f.read()
drawing = True


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
    ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')

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
    x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

    # Create minor ticks placed at each integer to enable drawing of minor grid
    # lines: note that this has no effect in this example with ticks_frequency=1
    ax.set_xticks(np.arange(xmin, xmax+1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax+1), minor=True)

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
    number = int(no_list.count(",")/no_list.count("["))
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
    heights = []
    bad_points = []
    good_faces = []
    for point in points:
        if point[2] not in heights:
            heights.append(point[2])
    top = max(heights)
    for i in range(len(points)):
        if not(points[i][2] >= top):
            bad_points.append(points[i])
    for face in faces:
        for point in face:
            if points[point] in bad_points:
                good_faces.append(face)
                break

    return good_faces


def worthless_points(points: list) -> list:
    heights = []
    bad_points = []
    for point in points:
        if point[2] not in heights:
            heights.append(point[2])
    top = max(heights)
    for i in range(len(points)):
        if not(points[i][2] >= top):
            bad_points.append(points[i])
    for point in bad_points:
        points.remove(point)
    return bad_points


def find_angle(a: list, mid: list, c: list) -> list:
    xa = a[0]
    ya = a[1]
    xmid = mid[0]
    ymid = mid[1]
    xc = c[0]
    yc = c[1]
  #  alpha = math.degrees(math.asin((ya - ymid) / math.sqrt(math.pow((xa - xmid), 2) + math.pow((ya - ymid), 2))))
    alpha = math.degrees(math.asin((-1)))
    if 90 > alpha > 0 > xa - xmid:
        alpha = 180 - alpha
    if 0 > alpha > -90 and xa - xmid < 0:
        alpha = 180 - alpha
    if alpha < 0:
        alpha = 360 - alpha



    beta = math.degrees(math.asin((yc - ymid) / math.sqrt(math.pow((xc - xmid), 2) + math.pow((yc - ymid), 2))))
    if 90 > beta > 0 > xa - xmid:
        beta = 180 - beta
    if 0 > beta > -90 and xa - xmid < 0:
        beta = 180 - beta
    if beta < 0:
        beta = 360 - beta

    print(alpha, beta)


if __name__ == '__main__':
    Points = list(find_points(f))
    Faces = list(find_faces(f))
    Faces = list(not_top_faces(Points, Faces))
    worthless_points(Points)
    find_angle(Points[0], Points[1], Points[2])
    if drawing:
        draw(Points)
        plt.show()
