from PIL import Image
import cv2
import numpy as np


# Open an Image
def open_image(path):
    new_image = cv2.imread(path)
    return new_image


# Save Image
def save_image(image, path):
    image.save(path, 'png')


def find_bounding_boxes(image):
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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


def new_image(height, width):
    blank_image = np.zeros((height, width, 3), np.uint8)
    return blank_image


def paste_image(image, pasted_image, x, y):
    height, width, channel = pasted_image.shape
    image[y:y+height, x:x+width] = pasted_image
    return image

im = open_image('testSimple.jpg')
contours = find_bounding_boxes(im)
avg_contour_w = 0
avg_contour_h = 0
height, width, channel = im.shape
min_size = max(height, width) * .01
max_size = min(height, width) * .5
print(min_size)
print(max_size)
letter_contours = []
for i, contour in enumerate(contours):
    # gets coordinates of Rect
    (x, y, w, h) = cv2.boundingRect(contour)
    if min_size < w < max_size or min_size < h < max_size:
        print('height:', h)
        print('width:', w)
        avg_contour_w += w
        avg_contour_h += h
        letter_contours.append(contour)

avg_contour_w /= len(letter_contours)
avg_contour_w = int(avg_contour_w)
avg_contour_h /= len(letter_contours)
avg_contour_h = int(avg_contour_h)
print(len(letter_contours))
cropped_letters = []
# we will crop each letter and scale it so they are uniform
for letter_contour in letter_contours:
    cropped_letter = crop_image(im, letter_contour)
    cropped_and_scaled_letter = scale_image(cropped_letter, avg_contour_h, avg_contour_w, min_size)
    cropped_letters.append(cropped_and_scaled_letter)

final_image = new_image(height, width)

(x2, y2, w, h) = cv2.boundingRect(letter_contours[0])
for cropped_letter, original_contour in zip(cropped_letters,letter_contours):
    (x, y, w, h) = cv2.boundingRect(original_contour)
    final_image = paste_image(final_image, cropped_letter, x, y2)


cv2.imwrite('testWrite.jpg', final_image)

'''
for contour in letter_contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    # if the contour is sufficiently large, it must be a digit
    x1 = x + w
    y1 = y + h
    # print(x,x1,y,y1)
    # Drawing the selected contour on the original image
    cv2.rectangle(im, (x, y), (x1, y1), (0, 255, 0), 2)
print('finished all')
print('done')
'''
