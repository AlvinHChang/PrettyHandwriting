from backend.image import Image
import cv2
import numpy as np
from backend.jenksPartition import getJenksBreaks


class Handwriting():
    def __init__(self, filename, input_folder, output_folder, file_extension, possible_lines=10, min_scalar=.01, max_scalar=.5, scaling_range_ratio=1.5):
        self.filename = filename
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.file_extension = file_extension
        self.file = self.filename + self.file_extension
        self.contours = None
        self.cropped_letters = None
        self.centroids = None
        self.line_placements = None
        self.min_scalar = min_scalar
        self.max_scalar = max_scalar
        self.scaling_range_ratio = scaling_range_ratio
        self.possible_lines = possible_lines
        self.avg_contour_h, self.avg_contour_w = 0, 0

        self.image_object = Image.open_image(self.input_folder + self.file)
        self.height, self.width, self.channel = self.image_object.shape

    @staticmethod
    def filter_contour_by_size(contours, min_size, max_size):
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

    def print_bounding_boxes(self, contours):
        copied_image = Image.copy_image(self.image_object)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            # if the contour is sufficiently large, it must be a digit
            x1 = x + w
            y1 = y + h
            # print(x,x1,y,y1)
            # Drawing the selected contour on the original image
            cv2.rectangle(copied_image, (x, y), (x1, y1), (0, 255, 0), 2)
        Image.save_image(self.output_folder + self.filename + 'Bounding' + self.file_extension, copied_image)

    def find_bounding_boxes_and_crop(self, should_print_bounding_boxes=False):
        # img = cv2.medianBlur(image, 5)

        imgray = cv2.cvtColor(self.image_object,cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        min_size = max(self.height, self.width) * self.min_scalar
        max_size = min(self.height, self.width) * self.max_scalar
        # filter one, removes anything that is too extreme
        filter_one_contour, avg_contour_h, avg_contour_w = Handwriting.filter_contour_by_size(contours, min_size, max_size)
        max_size = max(avg_contour_h, avg_contour_w) * self.scaling_range_ratio
        min_size = min(avg_contour_h, avg_contour_w) / self.scaling_range_ratio
        # filter two, removes anything that is too small or big to be relevant
        letter_contours_bounding_box, avg_contour_h, avg_contour_w = Handwriting.filter_contour_by_size(filter_one_contour, min_size, max_size)

        self.avg_contour_h, self.avg_contour_w = avg_contour_h, avg_contour_w
        self.contours = letter_contours_bounding_box
        if should_print_bounding_boxes:
            self.print_bounding_boxes(contours)
        cropped_letters = self.crop_and_scale(letter_contours_bounding_box, avg_contour_h, avg_contour_w, min_size)
        return letter_contours_bounding_box, cropped_letters

    def crop_and_scale(self, letter_contours, avg_contour_h, avg_contour_w, min_size):
        cropped_letters = []
        # we will crop each letter and scale it so they are uniform
        for letter_contour in letter_contours:
            cropped_letter = Image.crop_image(self.image_object, letter_contour)
            cropped_and_scaled_letter = Image.scale_image(cropped_letter, avg_contour_h, avg_contour_w, min_size)
            cropped_letters.append(cropped_and_scaled_letter)
        self.cropped_letters = cropped_letters
        return cropped_letters

    def find_centroids(self, should_print_centroids):
        """
        Find the centroid (moment-wise) of each contour, this allows a good sense of where the word belongs
        :param should_print_centroids: boolean whether to print the centroids
        :return: array of centroids
        """
        centroids = []
        for c in self.contours:
            # calculate moments for each contour
            M = cv2.moments(c)

            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
        if should_print_centroids:
            self.print_centroids(centroids)
        self.centroids = centroids
        return centroids

    def print_centroids(self, centroids):
        copied_image = Image.copy_image(self.image_object)
        for centroid in centroids:
            cX, cY = centroid
            cv2.circle(copied_image, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(copied_image, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # display the image
        Image.save_image(self.output_folder + self.filename + 'Centroid' + self.file_extension, copied_image)

    def print_line_placements(self, breaks):
        copied_image = Image.copy_image(self.image_object)
        for placement in breaks:
            # the shape is finding the width and dividing by two to get to middle
            cv2.circle(copied_image, (int(self.width / 2), int(placement)), 5, (255, 255, 255), -1)
        Image.save_image(self.output_folder + self.filename + 'Lines' + self.file_extension, copied_image)

    def find_line_placements(self, should_print_line_placements):

        # a list of cY so we can cluster them, essentially converts the centroids for 1d clustering
        centroids_flattened_to_y = [cY for (cX, cY) in self.centroids]

        # this gets the placement where the lines should go
        line_placements = getJenksBreaks(centroids_flattened_to_y, self.possible_lines)

        if should_print_line_placements:
            self.print_line_placements(line_placements)
        self.line_placements = line_placements
        return line_placements

    @staticmethod
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

    def print_final(self):
        final_image = Image.new_similar_image(self.image_object)

        for cropped_letter, original_contour_bounding_box in zip(self.cropped_letters, self.contours):
            x, y = self.distribute_letter_to_line(self.line_placements, original_contour_bounding_box)
            # so we can offset to correct height
            y -= self.avg_contour_h / 2
            y = int(y)
            final_image = Image.paste_image(final_image, cropped_letter, x, y)

        Image.save_image(self.output_folder + self.filename + 'Final' + self.file_extension, final_image)


