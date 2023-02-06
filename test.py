import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('26.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use Canny Edge Detection to find edges
edges = cv2.Canny(gray, 50, 150)

# Use Hough Transform to detect lines
for threshold in range(400, 100, -5):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, maxLineGap=100)
    print(threshold)
    if not lines is None and len(lines) > 20:
        break

# same thing but now we get 60 lines
for threshold in range(400, 100, -5):
    lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, maxLineGap=100)
    print(threshold)
    if len(lines) > 60:
        break

# Filter out the small lines
# lines = [line for line in lines if np.sqrt((line[0][0] - line[0][2]) ** 2 + (line[0][1] - line[0][3]) ** 2) > 50]

# Create a copy of the image to draw the lines on
line_image = np.copy(image)

# Get the endpoints of all the lines
endpoints = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    endpoints.append([[x1, y1], [x2, y2]])

endpoints2 = []
for line in lines2:
    x1, y1, x2, y2 = line[0]
    endpoints2.append([[x1, y1], [x2, y2]])

# Differentiate between horizontal and vertical lines
horizontal_lines = []
vertical_lines = []
for endpoint in endpoints:
    x1, y1 = endpoint[0]
    x2, y2 = endpoint[1]
    if abs(x2 - x1) > abs(y2 - y1):
        horizontal_lines.append(endpoint)
    else:
        vertical_lines.append(endpoint)

# in endpoints2 we only consider horizontal lines and we add to a different list
horizontal_lines2 = []
for endpoint in endpoints2:
    x1, y1 = endpoint[0]
    x2, y2 = endpoint[1]
    if abs(x2 - x1) > abs(y2 - y1):
        horizontal_lines2.append(endpoint)


horizontal_lines_orig = np.copy(horizontal_lines)
# Consider only extreme lines
highest_horizontal_line = max(horizontal_lines, key=lambda x: x[0][1])
lowest_horizontal_line = min(horizontal_lines, key=lambda x: x[0][1])
leftmost_vertical_line = min(vertical_lines, key=lambda x: x[0][0])
rightmost_vertical_line = max(vertical_lines, key=lambda x: x[0][0])

horizontal_lines = [highest_horizontal_line, lowest_horizontal_line]
vertical_lines = [leftmost_vertical_line, rightmost_vertical_line]

# Draw the horizontal lines in red
for endpoint in horizontal_lines:
    x1, y1 = endpoint[0]
    x2, y2 = endpoint[1]
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Draw the vertical lines in green
for endpoint in vertical_lines:
    x1, y1 = endpoint[0]
    x2, y2 = endpoint[1]
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

intersections = []
# Get intersection points and draw them in blue
for horizontal_line in horizontal_lines:
    for vertical_line in vertical_lines:
        x1, y1 = horizontal_line[0]
        x2, y2 = horizontal_line[1]
        x3, y3 = vertical_line[0]
        x4, y4 = vertical_line[1]
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        intersections.append([x, y])
        cv2.circle(line_image, (int(x), int(y)), 2, (255, 0, 0), 2)

# Correct order for the transform
intersections = [intersections[0], intersections[1], intersections[3], intersections[2]]
# mirro the image
intersections = [intersections[0], intersections[3], intersections[2], intersections[1]]
# rotate the image
intersections = [intersections[1], intersections[2], intersections[3], intersections[0]]
# # rotate the image
# intersections = [intersections[1], intersections[2], intersections[3], intersections[0]]
# # rotate again
# intersections = [intersections[1], intersections[2], intersections[3], intersections[0]]

# Perspective transform from intersections to a rectangle
src = np.array(intersections, dtype='float32')

# Define the size of the transformed image
x1, y1 = 1000, 1000

# Define the four corners of the parallelogram after the transform
dst = np.array([[0, 0], [x1, 0], [x1, y1], [0, y1]], dtype='float32')

# Calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)

# Put image limits in an array
image_limits = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]], dtype='float32')

# Transform those limits using the perspective transform matrix
transformed_limits = cv2.perspectiveTransform(image_limits.reshape(-1, 1, 2), M)

# Get bounding box of the transformed limits as points in an array
x_min = int(min(transformed_limits[:, 0, 0]))
x_max = int(max(transformed_limits[:, 0, 0]))
y_min = int(min(transformed_limits[:, 0, 1]))
y_max = int(max(transformed_limits[:, 0, 1]))

bounding_box = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype='float32')

# Use the inverse perspective transform to get the bounding box in the original image
bounding_box_orig = cv2.perspectiveTransform(bounding_box.reshape(-1, 1, 2), np.linalg.inv(M))

# Get new perspective transform matrix using your bounding box orig
src = np.array(bounding_box_orig, dtype='float32')
dst = np.array([[0, 0], [x1, 0], [x1, y1], [0, y1]], dtype='float32')
M = cv2.getPerspectiveTransform(src, dst)

# Apply the perspective transform to the image
warped = cv2.warpPerspective(image, M, (x1, y1))

# # Add to the line image all horizontal lines
# for endpoint in horizontal_lines_orig:
#     x1, y1 = endpoint[0]
#     x2, y2 = endpoint[1]
#     cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Get all horizontal lines in the transformed frame
transformed_horizontal_lines = []
for endpoint in horizontal_lines_orig:
    x1, y1 = endpoint[0]
    x2, y2 = endpoint[1]
    transformed_horizontal_lines.append(cv2.perspectiveTransform(np.array([[x1, y1], [x2, y2]], dtype='float32').reshape(-1, 1, 2), M))

# Add horizontal lines 2 as well
for endpoint in horizontal_lines2:
    x1, y1 = endpoint[0]
    x2, y2 = endpoint[1]
    transformed_horizontal_lines.append(cv2.perspectiveTransform(np.array([[x1, y1], [x2, y2]], dtype='float32').reshape(-1, 1, 2), M))


# Get all vertical distances between horizontal lines (considering all possibilities)
vertical_distances = []
for i in range(len(horizontal_lines_orig)):
    for j in range(i + 1, len(horizontal_lines_orig)):
        vertical_distances.append(abs(horizontal_lines_orig[i][0][1] - horizontal_lines_orig[j][0][1]))
# Get histogram of vertical distances
hist = np.histogram(vertical_distances, bins=50)
# Get first non empty bin other than the first one
val = None
for i in range(1, len(hist[0])):
    if hist[0][i] > 0:
        val = hist[1][i]
        break

# Add ten lines above ther first horizontal line in thransformed horizontal lines spaced of val
lines_to_add = []
for i in range(-15, 20):
    x1, y1 = transformed_horizontal_lines[0][0][0]
    x2, y2 = transformed_horizontal_lines[0][1][0]
    lines_to_add.append([[x1, y1 - (i + 1) * val], [x2, y2 - (i + 1) * val]])

# Add warped image trnasformed horizontal lines
for endpoint in transformed_horizontal_lines:
    x1, y1 = endpoint[0][0]
    x2, y2 = endpoint[1][0]
    cv2.line(warped, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# # Add lines to add
# for endpoint in lines_to_add:
#     x1, y1 = endpoint[0]
#     x2, y2 = endpoint[1]
#     cv2.line(warped, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# Plot the original image and the image with the differentiated lines
# plt.subplot(121)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("Original Image")
plt.subplot(121)
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
plt.title("Transformed Image")

plt.subplot(122)
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.title("Image with lines highlighted")
plt.show()

# Plot edges image