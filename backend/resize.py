from backend.jenksPartition import getJenksBreaks
from PIL import Image
import numpy as np
import cv2
import random


# Open an Image
def open_image(path):
    new_image = cv2.imread(path)
    return new_image


# Save Image
def save_image(image, path):
    image.save(path, 'png')


def print_bounding_boxes(image, contours):
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # if the contour is sufficiently large, it must be a digit
        x1 = x + w
        y1 = y + h
        # print(x,x1,y,y1)
        # Drawing the selected contour on the original image
        cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 2)
    cv2.imwrite(output_file_name + 'Bounding' + file_extension, image)


def find_bounding_boxes(image):
    # this method is called Adaptive Thresholding, it accounts for lighting differences
    # img = cv2.medianBlur(image, 5)

    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
    th3 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7)

    cv2.imwrite(output_file_name + 'Gray2' + file_extension, th2)
    cv2.imwrite(output_file_name + 'Gray3' + file_extension, th3)

    contours, hierarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Create a new image with the given size
def create_image(i, j):
    image = Image.new("RGB", (i, j), "white")
    return image


# Get the pixel from the given image
def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
        return None

    # Get Pixel
    pixel = image.getpixel((i, j))
    return pixel


# finds the average shade of the image so we can compare it to text
def get_average_shade(image):
    width, height = image.size
    average_r = 0
    average_g = 0
    average_b = 0
    for column in range(width):
        for pixel in range(height):
            pixel_color = get_pixel(image, column, pixel)
            average_r += pixel_color[0]
            average_g += pixel_color[1]
            average_b += pixel_color[2]
    image_pixels = width * height
    average_r /= image_pixels
    average_g /= image_pixels
    average_b /= image_pixels
    return average_r, average_b, average_g


def crop_image(image, contour):
    (x, y, w, h) = cv2.boundingRect(contour)
    crop_image = image[y:y + h, x:x + w]
    return crop_image


def scale_image(rescaling_image, new_h, new_w, min_size):
    height, width, channel = rescaling_image.shape
    # letter is probably i or l, their width is too thin
    if width < min_size:
        # we only rescale the height in that case
        rescaled_image = cv2.resize(rescaling_image, (width, new_h))
    else:
        rescaled_image = cv2.resize(rescaling_image, (new_w, new_h))
    return rescaled_image


def new_similar_image(image):
    height, width, channel = image.shape
    center_height, center_width = int(height/2), int(width/2)
    for iteration in range(1, 10):
        background_color = image[center_height, center_width]
        background_color = int(np.mean(background_color))
        # 127 is mean gray, less than means it's too dark
        if background_color < MEAN_GRAY:
            # we will try to select more towards the middle
            center_width = random.randint(int(width/4), int(3*width/4))
            center_height = random.randint(int(height/4), int(3*height/4))
        # else is acceptable value
        else:
            break

    blank_image = np.full((height, width, 3), background_color, np.uint8)
    return blank_image


def paste_image(image, pasted_image, x, y):
    height, width, channel = pasted_image.shape
    image[y:y+height, x:x+width] = pasted_image
    return image


def find_centroids(contours):
    """
    Find the centroid (moment-wise) of each contour, this allows a good sense of where the word belongs
    :param contours: array of contours provided by cv2 package
    :return: array of centroids
    """
    centroids = []
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append((cX, cY))
    return centroids


def print_centroids(image, centroids):
    for centroid in centroids:
        cX, cY = centroid
        cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(image, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # display the image
    cv2.imwrite(output_file_name + 'Moments' + file_extension, image)


def print_line_placements(image, breaks):
    for placement in breaks:
        # the shape is finding the width and dividing by two to get to middle
        cv2.circle(image, (int(image.shape[1] / 2), int(placement)), 5, (255, 255, 255), -1)
    cv2.imwrite(output_file_name + 'Lines' + file_extension, image)


def filter_contour(contours, min_size, max_size):
    """
    Filters the contours given min_size and max_size, will go through if EITHER height or width passes
    :param contours: Array of contours
    :param min_size: Minimum size that a dimension can pass
    :param max_size: Maximum size that a dimension can pass
    :return: A new array that contains filtered contours, average height, average width
    """
    filtered_contours = []
    avg_contour_w = 0
    avg_contour_h = 0
    for contour in contours:
        # gets coordinates of Rect
        (x, y, w, h) = cv2.boundingRect(contour)
        if min_size < w < max_size or min_size < h < max_size:
            avg_contour_w += w
            avg_contour_h += h
            filtered_contours.append(contour)
    avg_contour_w /= len(filtered_contours)
    avg_contour_w = int(avg_contour_w)
    avg_contour_h /= len(filtered_contours)
    avg_contour_h = int(avg_contour_h)
    return filtered_contours, avg_contour_h, avg_contour_w


def distribute_letter_to_line(line_placements, contour_bounding_box):
    """
    Places the letter on the correct line that is belongs to in the original text file
    :param line_placements: array of possible line placement heights
    :param contour_bounding_box: the bounding box of the original letter
    :return: x, y coordinate of the letter
    """
    (x, y, w, h) = cv2.boundingRect(contour_bounding_box)
    approximating_y_error = 10000
    best_line_placement = None
    for line_placement in line_placements:
        # this finds the best approximate line it should go on if it doesn't find a good placement
        closest_placement = min(abs(y - line_placement), abs(y+h - line_placement))
        if closest_placement < approximating_y_error:
            approximating_y_error = closest_placement
            best_line_placement = line_placement
        # the letter's bounding box fits the line
        if y < line_placement < y + h:
            return x, line_placement
    return x, best_line_placement


filename = 'testNoteBright'
input_file_name = 'input/' + filename
output_file_name = 'output/' + filename
file_extension = '.jpg'
file = input_file_name + file_extension
# the range to count as too extreme
scaling_range_ratio = 1.5
possible_lines = 10
should_print_line_placements = False
should_print_bounding_boxes = False
should_print_centroids = False
# mean value of color gray
MEAN_GRAY = 127

im = open_image(file)
contours = find_bounding_boxes(im)
height, width, channel = im.shape
min_size = max(height, width) * .01
max_size = min(height, width) * .5
print(min_size)
print(max_size)
# print_bounding_boxes(im, contours)
# filter one, removes anything that is too extreme
filter_one_contour, avg_contour_h, avg_contour_w = filter_contour(contours, min_size, max_size)
max_size = max(avg_contour_h, avg_contour_w) * scaling_range_ratio
min_size = min(avg_contour_h, avg_contour_w) / scaling_range_ratio
# filter two, removes anything that is too small or big to be relevant
letter_contours_bounding_box, avg_contour_h, avg_contour_w = filter_contour(filter_one_contour, min_size, max_size)
if should_print_bounding_boxes:
    print_bounding_boxes(im, letter_contours_bounding_box)

cropped_letters = []
# we will crop each letter and scale it so they are uniform
for letter_contour in letter_contours_bounding_box:
    cropped_letter = crop_image(im, letter_contour)
    cropped_and_scaled_letter = scale_image(cropped_letter, avg_contour_h, avg_contour_w, min_size)
    cropped_letters.append(cropped_and_scaled_letter)

centroids = find_centroids(letter_contours_bounding_box)

if should_print_centroids:
    print_centroids(im, centroids)
# a list of cY so we can cluster them, essentially converts the centroids for 1d clustering
centroids_flattened_to_y = [cY for (cX, cY) in centroids]

# this gets the placement where the lines should go
line_placements = getJenksBreaks(centroids_flattened_to_y, possible_lines)

if should_print_line_placements:
    print_line_placements(im, line_placements)

final_image = new_similar_image(im)

for cropped_letter, original_contour_bounding_box in zip(cropped_letters, letter_contours_bounding_box):
    x, y = distribute_letter_to_line(line_placements, original_contour_bounding_box)
    # so we can offset to correct height
    y -= avg_contour_h / 2
    y = int(y)
    final_image = paste_image(final_image, cropped_letter, x, y)


cv2.imwrite(output_file_name + 'Final' + file_extension, final_image)
print('printed')

