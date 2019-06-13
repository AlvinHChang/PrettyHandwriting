from backend.image import Image
from backend.handwriting import Handwriting
import numpy as np
import cv2
import random



filename = 'testNoteBright'
root = 'backend/'
input_folder = root + 'input/'
output_folder = root + 'output/'
file_extension = '.jpg'
document = Handwriting(filename, input_folder, output_folder, file_extension)
# the range to count as too extreme
scaling_range_ratio = 1.5
possible_lines = 10
should_print_line_placements = True
should_print_bounding_boxes = True
should_print_centroids = True

letter_contours, cropped_letters = document.find_bounding_boxes_and_crop(should_print_bounding_boxes)


centroids = document.find_centroids(should_print_centroids)

line_placements = document.find_line_placements(should_print_line_placements)

document.print_final()


