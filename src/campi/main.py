import cv2 as cv
import timeit
from threading import Thread
import picamera2
from typing import List, Union
from queue import Queue
import yaml
from utils import plot_image_channels
from calibration import CameraCalibrator, CalibrationLoader
from gpios import readVisibleLux, main_call
import argparse
from time import sleep


#     sleep(1)
#     pic = cam.capture_array()
#     cv.imwrite('data/pictures/image.jpg', pic)
#     print(pic.shape)
#     plot_image_channels(pic, titles=['Red', 'Green', 'Blue'], max_channels=3)
#     print("Image captured and saved as 'image.jpg'")

# class ResultThread(Thread):
#     def __init__(self, target):
#         super().__init__()
#         self._target = target
#         self.result = None

#     def run(self):
#         self.result = self._target()

#     def camera():
#         cam = picamera2.Picamera2(camera_num=0)
#         cam.start()
#         img = cam.capture_array()
#         cam.stop()
#         return img

#     def light():
#         return readVisibleLux()

class Camera(Thread):
    def __init__(self, camera: int = 0):
        super().__init__()
        self.image_queue = Queue()
        self.light_queue = Queue()
        self.cam = picamera2.Picamera2(camera_num=0)

    def switch_off(self):
        self.cam.stop()

    def take_picture_threading(self):
        self.cam.start()
        img = self.cam.capture_array()
        self.cam.stop()
        self.image_queue.put(img)

    def take_picture(self):
        self.cam = picamera2.Picamera2(camera_num=0)
        self.cam.start()
        img = self.cam.capture_image()
        self.cam.stop()
        return img

    def read_light(self):
        self.light_queue.put(readVisibleLux())

    def measure(self):
        start = timeit.default_timer()
        t1 = Thread(target=self.take_picture_threading)
        t2 = Thread(target=self.read_light)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        stop = timeit.default_timer()
        print(f"Time taken: {stop - start} seconds")

class TwoCameras(Thread):
    def __init__(self):
        super().__init__()

        #Camera 0
        self.image0 = Queue()
        self.cam0 = picamera2.Picamera2(camera_num=0)

        #Camera 1
        self.image1 = Queue()
        self.cam1 = picamera2.Picamera2(camera_num=1)

        self.light_measurement = Queue()
        

    def take_picture_threading(self, camera:picamera2.Picamera2, image_queue: Queue):
        start = timeit.default_timer()
        camera.start()
        img = camera.capture_array()
        camera.stop()
        image_queue.put(img)
        stop = timeit.default_timer()
        print(f"Image capture time taken: {stop - start} seconds")


    def read_light(self):
        start = timeit.default_timer()
        self.light_measurement.put(readVisibleLux())
        stop = timeit.default_timer()
        print(f"Light measurement time taken: {stop - start} seconds")

    def measure(self):
        start = timeit.default_timer()

        threads = [
            Thread(target=self.take_picture_threading, args=(self.cam0, self.image0)),
            Thread(target=self.take_picture_threading, args=(self.cam1, self.image1)),
            Thread(target=self.read_light),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        stop = timeit.default_timer()
        print(f"Time taken: {stop - start} seconds")




def open_calibration_settings():
    calibrator = CameraCalibrator(camera=0,
                                chessboard_size=config['chessboard_size'],
                                square_size=config['square_size'])
    decision = input("Start to calibrate? (y/n): ")
    if decision.lower() == 'y':
        calibrator.calibration_managment()
    else:
        print("Calibration aborted.")
        return 0



if __name__ == "__main__":
    with open('src/campi/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        print("Configuration loaded:", config)
        if config['calibrate_true']:
            open_calibration_settings()
        try:
            calibrator = CalibrationLoader()    
            calibrator.load_calibration()
            
        except FileNotFoundError:
            print("Calibration file not found. Please run calibration first.")
            open_calibration_settings()
        except Exception as e:
            print(f"Error loading calibration: {e}")
            open_calibration_settings()
        
        


    # cam = TwoCameras()
    # for i in range(1):
    #     cam.measure()
    # cam.image0 = cam.image0.get()
    # cam.image1 = cam.image1.get()

    # rgb = cv.cvtColor(cam.image0, cv.COLOR_BGR2RGB)
    # rgbir = cv.cvtColor(cam.image1, cv.COLOR_BGR2RGB)

    # avg_r_rgb = rgb[:, :, 0].mean()
    # avg_r_rgbir = rgbir[:, :, 0].mean()
    # intensity_r_rgb = rgb[:, :, 0].sum()
    # intensity_r_rgbir = rgbir[:, :, 0].sum()
    # max_r_rgb = rgb[:, :, 0].max()
    # max_r_rgbir = rgbir[:, :, 0].max()

    # total_intensity_rgb = rgb[:, :, :].sum()
    # total_intensity_rgbir = rgbir[:, :, :].sum()

    # print("Average Red channel value for RGB camera:", avg_r_rgb)
    # print("Max Red channel value for RGB camera:", max_r_rgb)
    # print("Average Red channel value for RGBIR camera:", avg_r_rgbir)
    # print("Max Red channel value for RGBIR camera:", max_r_rgbir)
    # # print("Difference in Red channel values:", avg_r_rgbir - avg_r_rgb)
    # print("Total intensity of Red channel for RGB camera:", intensity_r_rgb/total_intensity_rgb)
    # print("Total intensity of Red channel for RGBIR camera:", intensity_r_rgbir/total_intensity_rgbir)

    # print("Ratio of Red channel values:", avg_r_rgb/avg_r_rgbir)
    # light =  cam.light_measurement.get()

    # print("Light measurement:", light[2]/light[1])


    # cv.imwrite('/home/jacob/campi/src/data/image0.jpg', rgb)
    # cv.imwrite('/home/jacob/campi/src/data/image1.jpg', rgbir)

    # # print("Image0 shape:", cam.image0.get().shape)
    # # print("Image1 shape:", cam.image1.get().shape)
    # # print("Light reading:", cam.light_measurement.get())
    # cv.waitKey(0)





