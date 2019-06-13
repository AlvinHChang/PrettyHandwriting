import numpy as np
import cv2
import random


class Image():

    # Open an Image
    @staticmethod
    def open_image(path):
        return cv2.imread(path)

    @staticmethod
    def save_image(filename, image):
        cv2.imwrite(filename, image)

    @staticmethod
    def paste_image(tgt_image, src_image, x, y):
        height, width, channel = src_image.shape
        tgt_image[y:y+height, x:x+width] = src_image
        return tgt_image

    @staticmethod
    def scale_image(src_image, new_h, new_w, min_size):
        height, width, channel = src_image.shape
        # letter is probably i or l, their width is too thin
        if width < min_size:
            # we only rescale the height in that case
            rescaled_image = cv2.resize(src_image, (width, new_h))
        else:
            rescaled_image = cv2.resize(src_image, (new_w, new_h))
        return rescaled_image

    @staticmethod
    def crop_image(src_image, bound):
        (x, y, w, h) = cv2.boundingRect(bound)
        cropped_image = src_image[y:y + h, x:x + w]
        return cropped_image

    @staticmethod
    def new_image(height, width, color=255):
        blank_image = np.full((height, width, 3), color, np.uint8)
        return blank_image

    @staticmethod
    def copy_image(image):
        new_image = image.copy()
        return new_image

    @staticmethod
    def new_similar_image(image):
        # mean value of color gray
        MEAN_GRAY = 127
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

