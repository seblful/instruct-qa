import math


def get_points_bbox(points):
    minmax = [None, None, None, None]

    for i, num in enumerate(points):
        pos = round(i / 2) * 2 - i

        if pos == 0:
            # Calculate min and max X
            if minmax[0] is None or minmax[0] >= num:
                minmax[0] = num
            if minmax[2] is None or minmax[2] <= num:
                minmax[2] = num
        elif pos == 1:
            # Calculate min and max Y
            if minmax[1] is None or minmax[1] >= num:
                minmax[1] = num
            if minmax[3] is None or minmax[3] <= num:
                minmax[3] = num

    return minmax


def get_rect_bbox(x, y, width, height, angle):
    angle_rad = math.radians(angle)

    def rotate(x1, y1):
        return [
            (x1 - x) * math.cos(angle_rad) -
            (y1 - y) * math.sin(angle_rad) + x,
            (x1 - x) * math.sin(angle_rad) +
            (y1 - y) * math.cos(angle_rad) + y,
        ]

    rx1, ry1, rx2, ry2 = get_points_bbox([
        x,
        y,
        *rotate(x + width, y),
        *rotate(x + width, y + height),
        *rotate(x, y + height),
    ])

    return rx1, ry1, rx2 - rx1, ry2 - ry1
