from math import cos, pi, sin, sqrt, tan
from statistics import median
import numpy as np
import cv2 as cv

class Line:
    _rho_diff_threshold = 80.0
    _theta_diff_threshold = 0.3
    _distance_threshold = 40

    def __init__(self, rho: float, theta: float) -> 'Line':
        self.rho = rho
        self.theta = theta

    def is_similar(self, to_compare: 'Line', frame) -> bool:
        A, B = _get_line_frame_intersection_points(self, frame)
        C, D = _get_line_frame_intersection_points(to_compare, frame)
        AB = LineSegment(A, B)
        CD = LineSegment(C, D)

        dist_A_to_CD = self._distance_from_point_to_line_segment(A, CD)
        dist_B_to_CD = self._distance_from_point_to_line_segment(B, CD)
        dist_C_to_AB = self._distance_from_point_to_line_segment(C, AB)
        dist_D_to_AB = self._distance_from_point_to_line_segment(D, AB)

        return (dist_A_to_CD < self._distance_threshold
                and dist_B_to_CD < self._distance_threshold
                and dist_C_to_AB < self._distance_threshold
                and dist_D_to_AB < self._distance_threshold)

    def _distance_from_point_to_line_segment(self, point: 'tuple[int, int]', line_segment: 'LineSegment') -> float:
        x0, y0 = point
        x1, y1 = line_segment.start_point
        x2, y2 = line_segment.end_point
        return abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / sqrt((x2-x1)**2 + (y2-y1)**2)


class LineSegment:
    def __init__(self, start_point: 'tuple[int, int]', end_point: 'tuple[int, int]') -> 'LineSegment':
        self.start_point = start_point
        self.end_point = end_point

    def get_length(self):
        return sqrt((self.end_point[0] - self.start_point[0]) ** 2 + (self.end_point[1] - self.start_point[1]) ** 2)

    def flip(self) -> 'LineSegment':
        temp = self.start_point
        self.start_point = self.end_point
        self.end_point = temp
        return self

    def __str__(self) -> str:
        return f'start_point: {self.start_point}, end_point: {self.end_point}'

    def __repr__(self) -> str:
        return self.__str__()


class LineProcessor:
    def __init__(self, box_size=20, pixels_threshold=20, min_line_segment_size=3, min_line_segment_hole_size=2):
        self._BOX_SIZE = box_size
        self._PIXELS_THRESHOLD = pixels_threshold
        self._MIN_LINE_SEGMENT_SIZE = min_line_segment_size
        self._MIN_LINE_SEGMENT_HOLE_SIZE = min_line_segment_hole_size

    def _get_from_houghlines(self, hough_lines) -> 'list[Line]':
        if hough_lines is None:
            return []
        lines = []
        for line in hough_lines:
            rho, theta = line[0]
            lines.append(Line(rho, theta))
        return lines

    def _merge_lines(self, lines: 'list[Line]', frame) -> 'list[Line]':
        similar_lines: dict[Line, list[Line]] = {}
        for line in lines:
            found_similar = False
            for group_leading_line, grouped_lines in similar_lines.items():
                if line.is_similar(group_leading_line, frame):
                    found_similar = True
                    grouped_lines.append(line)
                    break
            if not found_similar:
                similar_lines[line] = [line]

        merged_lines = []
        for grouped_lines in list(similar_lines.values()):
            merged_lines.append(self._get_median_line(grouped_lines))
        return merged_lines

    def get_intersection_points(self, lines: 'list[Line]', frame) -> 'list[tuple[int, int]]':
        intersection_points = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                intersection_points.append(self._get_intersection_point(lines[i], lines[j], frame))
        return intersection_points
    
    def _get_intersection_point(self, line_one, line_two, frame):
        #https://stackoverflow.com/a/49645641
        segment_one = _get_line_frame_intersection_points(line_one, frame)
        segment_two = _get_line_frame_intersection_points(line_two, frame)
        line_one_start_homo = [segment_one[0][0], segment_one[0][1], 1]
        line_one_end_homo = [segment_one[1][0], segment_one[1][1], 1]
        line_two_start_homo = [segment_two[0][0], segment_two[0][1], 1]
        line_two_end_homo = [segment_two[1][0], segment_two[1][1], 1]
        
        line_one_homo = np.cross(line_one_start_homo, line_one_end_homo)
        line_two_homo = np.cross(line_two_start_homo, line_two_end_homo)
        intersection_homo = np.cross(line_one_homo, line_two_homo)
        return (intersection_homo[0]/intersection_homo[-1], intersection_homo[1]/intersection_homo[-1])
        



    # Return true if line segments AB and CD intersect
    def _intersect(self, A, B, C, D) -> bool:
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self. _ccw(A, B, D)
    # https://stackoverflow.com/a/9997374

    def _get_median_line(self, lines: 'list[Line]') -> 'Line':
        has_to_convert = self._max_theta_diff(
            lines) > 2*Line._theta_diff_threshold
        rhos = []
        thetas = []
        for line in lines:
            rho = line.rho
            theta = line.theta
            if(has_to_convert and theta > pi - Line._theta_diff_threshold):
                line = self._convert_to_comparable_form(line)
                rho, theta = line.rho, line.theta
            rhos.append(rho)
            thetas.append(theta)
        median_rho = median(rhos)
        median_theta = median(thetas)
        return self._convert_to_conventional_form(Line(median_rho, median_theta))
    
    def _max_theta_diff(self, lines: 'list[Line]') -> float:
        min_theta = lines[0].theta
        max_diff = 0
        for line in lines:
            if (abs(line.theta - min_theta) > max_diff):
                max_diff = abs(line.theta - min_theta)
            if (line.theta < min_theta):
                min_theta = line.theta
        return max_diff


    def _convert_to_comparable_form(self, line: 'Line') -> 'Line':
        rho, theta = line.rho, line.theta
        # if(rho < 0):
        theta -= pi
        rho *= -1
        return Line(rho, theta)

    def _convert_to_conventional_form(self, line: 'Line') -> 'Line':
        rho, theta = line.rho, line.theta
        if theta < 0:
            theta += pi
            rho *= -1
        return Line(rho, theta)


# HELPER FUNCTIONS

def _get_line_frame_intersection_points(line: 'Line', frame) -> 'list[tuple[int, int]]':
    max_x = int(frame.shape[1] - 1)
    max_y = int(frame.shape[0] - 1)
    rho, theta = line.rho, line.theta
    x_intercept = int(rho*cos(theta) - rho*sin(theta)/tan(theta - pi/2))
    y_intercept = int(rho*sin(theta) - rho*cos(theta)*tan(theta - pi/2))
    possible_frame_intersection_points = [
        (0, y_intercept),  # point is somewhere on the left side of the frame
        # point is somewhere on the right side of the frame
        (max_x, int(tan(theta - pi/2)*max_x + y_intercept)),
        (x_intercept, 0),  # point is somewhere on the top of the frame
        # point is somwhere on the bottom of the frame
        (int(1/tan(theta - pi/2)*max_y + x_intercept), max_y)
    ]
    possible_frame_intersection_points = _filter_out_of_frame(
        possible_frame_intersection_points, max_x, max_y)
    frame_intersection_points = _filter_same_points(
        possible_frame_intersection_points)[0:2]
    return frame_intersection_points


def _filter_out_of_frame(possible_frame_intersection_points: 'list[tuple[int, int]]', max_x, max_y) -> 'list[tuple[int, int]]':
    filtered_points = []
    for possible_point in possible_frame_intersection_points:
        if _is_within_frame(possible_point, max_x, max_y):
            filtered_points.append(possible_point)
    return filtered_points


def _filter_same_points(possible_frame_intersection_points: 'list[tuple[int, int]]') -> 'list[tuple[int, int]]':
    return list(set(possible_frame_intersection_points))


def _is_within_frame(point: 'tuple[int, int]', max_x, max_y) -> bool:
    x, y = point
    return x >= 0 and x <= max_x and y >= 0 and y <= max_y


def _get_intersection_point(line_one: 'Line', line_two: 'Line') -> 'tuple[int, int]':
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line_one.rho, line_one.theta
    rho2, theta2 = line_two.rho, line_two.theta
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return (x0, y0)