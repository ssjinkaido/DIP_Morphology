import cv2
from PIL import Image
from PIL import ImageChops
import matplotlib.pyplot as plt

thinning_structuring_elements = [[[0, 0, 0],
                                  [2, 255, 2],
                                  [255, 255, 255]],
                                 [[2, 0, 0],
                                  [255, 255, 0],
                                  [2, 255, 2]],
                                 [[255, 2, 0],
                                  [255, 255, 0],
                                  [255, 2, 0]],
                                 [[2, 255, 2],
                                  [255, 255, 0],
                                  [2, 0, 0]],
                                 [[255, 255, 255],
                                  [2, 255, 2],
                                  [0, 0, 0]],
                                 [[2, 255, 2],
                                  [0, 255, 255],
                                  [0, 0, 2]],
                                 [[0, 2, 255],
                                  [0, 255, 255],
                                  [0, 2, 255]],
                                 [[0, 0, 2],
                                  [0, 255, 255],
                                  [2, 255, 2]],
                                 ]


def open_image(filename):
    image = Image.open(filename)
    image = image.convert('1')  # binary image
    image.show()
    return image


def compare_element(image, element, x_offset, y_offset, image_pixels):
    pixels = [[image.getpixel((x_offset - 1, y_offset - 1)), image.getpixel((x_offset, y_offset - 1)),
               image.getpixel((x_offset + 1, y_offset - 1))],
              [image.getpixel((x_offset - 1, y_offset)), image.getpixel((x_offset, y_offset)),
               image.getpixel((x_offset + 1, y_offset))],
              [image.getpixel((x_offset - 1, y_offset + 1)), image.getpixel((x_offset, y_offset + 1)),
               image.getpixel((x_offset + 1, y_offset + 1))]]
    for x in range(0, 3):
        for y in range(0, 3):
            if element[y][x] == 2:
                continue
            elif pixels[y][x] != element[y][x]:
                return False
    image_pixels[x_offset, y_offset] = 0
    return


def images_equal(image1, image2):
    return ImageChops.difference(image1, image2).getbbox() is None


def thin(filename):
    last_image = open_image(filename)
    image = thin_helper(last_image.copy(), last_image.load())
    while not images_equal(image, last_image):
        print("iteration")
        last_image = image.copy()
        image = thin_helper(image, image.load())

    return last_image


def thin_helper(image, image_pixels):
    for element in thinning_structuring_elements:
        for i in range(1, image.size[0] - 1):
            for j in range(1, image.size[1] - 1):
                compare_element(image, element, i, j, image_pixels)
    return image.copy()


last_image = thin("kde.png")
img = cv2.imread("kde.png")
plt.subplot(121)
plt.title('Original image')
plt.imshow(img, cmap='gray')

plt.subplot(122)
plt.title('After Thinning')
plt.imshow(last_image, cmap='gray')

plt.show()
