import cv2 as cv
import numpy as np
from math import cos, pi, sin, sqrt
from lib_process_lines import LineProcessor

image_height = 640
image_width = 640

def white_pixel_in_neighbourhood(x, y, edges, neighbourhood_size=10):
    for i in range(-neighbourhood_size//2, neighbourhood_size//2 + 1):
        for j in range(-neighbourhood_size//2, neighbourhood_size//2 + 1):
            if edges[int(np.clip(y + i, 0, image_height)), int(np.clip(x + j, 0, image_width))] == 255:
                return True
    return False

def distance(point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Load image
image = cv.imread('test_images/box.jpg')
# Convert to grayscale and blur 
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray_blurred = cv.medianBlur(image_gray, 11, 0)
# Detect edges and find contours and filter out small contours
edges = cv.Canny(image_gray, 50, 150)
contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours_filtered = list(filter(lambda contour: cv.contourArea(contour) > 100, contours))

# Use contours and edges to only keep edges that are part of the outline of the box not the inside
contourMask = np.zeros_like(image_gray)
cv.drawContours(contourMask, contours_filtered, -1, 255, 1)
filtered_edges = cv.bitwise_and(edges, contourMask)

# Get intersection points of lines
line_processor = LineProcessor()
hough_lines = cv.HoughLines(filtered_edges, 1, np.pi/180, 25)
lines = line_processor._get_from_houghlines(hough_lines)
merged_lines = line_processor._merge_lines(lines, image)
intersections = line_processor.get_intersection_points(merged_lines, image)
# Filter out intersections that are outside the image or not close to a white pixel in the filtered edges mask
intersections_filtered = filter(lambda intersection: 0 <= intersection[0] < image.shape[1] and 0 <= intersection[1] < image.shape[0], intersections)
intersections_filtered = filter(lambda intersection: white_pixel_in_neighbourhood(intersection[0], intersection[1], filtered_edges), intersections_filtered)
intersections_filtered_int = map(lambda intersection: (int(intersection[0]), int(intersection[1])), intersections_filtered)
intersections_filtered_int_sorted = list(sorted(intersections_filtered_int, key=lambda intersection: intersection[1], reverse=True))
intersections_no_duplicates = [intersections_filtered_int_sorted[0]]

for i in range(len(intersections_filtered_int_sorted)):
    is_duplicate = False
    for j in range(len(intersections_no_duplicates)):
        if distance(intersections_filtered_int_sorted[i], intersections_no_duplicates[j]) < 15:
            is_duplicate = True

    if not is_duplicate:
        intersections_no_duplicates.append(intersections_filtered_int_sorted[i])
            
max_dist_intersection_midpoint = None
max_dist = 0
for i in range(3):
    intersection1 = intersections_no_duplicates[i]
    for j in range(i, 3):
        if distance(intersection1, intersections_no_duplicates[j]) > max_dist:
            max_dist = distance(intersection1, intersections_no_duplicates[j])
            max_dist_intersection_midpoint = ((intersection1[0] + intersections_no_duplicates[j][0])//2, (intersection1[1] + intersections_no_duplicates[j][1])//2)

for intersection in intersections_no_duplicates:
    cv.circle(image, intersection, 5, (255, 0, 0), 2)

cv.circle(image, max_dist_intersection_midpoint, 5, (0, 255, 0), 2)

for line in merged_lines:
    rho, theta = line.rho, line.theta
    a = cos(theta)
    b = sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv.imshow('Contours', image)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
