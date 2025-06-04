import cv2 as cv
from utils import plot_image_channels
from gpios import main_call
from time import sleep


# def main():
#     cam = Picamera2()
#     sleep(1)
#     pic = cam.capture_array()
#     cv.imwrite('data/pictures/image.jpg', pic)
#     print(pic.shape)
#     plot_image_channels(pic, titles=['Red', 'Green', 'Blue'], max_channels=3)
#     print("Image captured and saved as 'image.jpg'")

if __name__ == "__main__":
    main_call()