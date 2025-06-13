import cv2
import numpy as np
import picamera2
import os
import glob
import pickle
from time import sleep
from datetime import datetime

class CameraCalibrator:
    def __init__(self, camera: dict, chessboard_size: tuple[int], square_size=1.0, image_dir="src/data"):
        """
        Initialize the camera calibrator
        
        Args:
            chessboard_size: (width, height) of internal corners in chessboard
            square_size: Size of each square in real world units (e.g., mm, cm)
        """
        self.camera = camera
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.image_dir = os.path.join(image_dir, "calibration_images")
        self.test_image_dir = os.path.join(image_dir, "calibration_testing")
        self.calibration_file = os.path.join("src","campi","camera_calibration.pkl")

        for dir_path in [self.image_dir, self.test_image_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # cv2.TERM_CRITERIA_EPS = Stop the algorithm if the specified accuracy (epsilon) is reached
        # cv2.TERM_CRITERIA_MAX_ITER top the algorithm after a certain number of iterations (max_count).
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points on the chessboard (internal corners)
        # objp will be a 3D array of points in the chessboard coordinate system
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
    def capture_calibration_images(self, num_images=20):
        """
        Capture calibration images from camera
        
        Args:
            num_images: Number of calibration images to capture
            save_dir: Directory to save calibration images
        """
        # Create directory if it doesn't exist
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize camera
        cam = picamera2.Picamera2(camera_num=0)
        cam.start()
        if not cam.started:
            print("Error: Could not open camera")
            return False
            
        print(f"Starting calibration image capture...")
        print(f"Press SPACE to capture image, ESC to exit")
        print(f"Target: {num_images} images with detected chessboard")
        
        captured_count = 0
        
        while captured_count < num_images:
            frame = cam.capture_array()

            # if not ret:
            #     # not_detected_count += 1
            #     print("Error: Could not read frame")
            #     # sleep(2)
            #     # if not_detected_count >= 15:
            #     #     print("No frames detected for 15 attempts, exiting...")
            #     #     break
            #     # print(f"Retrying... ({not_detected_count}/15)")
            #     # continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret_corners, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            # Draw corners if found
            display_frame = frame.copy()
            if ret_corners:
                cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret_corners)
                print(f"Chessboard detected")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.image_dir, f"calibration_{timestamp}_{captured_count:02d}.jpg")
                cv2.imwrite(filename, frame)
                captured_count += 1
                sleep(0.5)  # Give some time before next capture
                # cv2.putText(display_frame, "Chessboard detected! Press SPACE to capture", 
                #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                print("No chessboard detected")
                sleep(0.5)
                # cv2.putText(display_frame, "No chessboard detected", 
                #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # cv2.putText(display_frame, f"Captured: {captured_count}/{num_images}", 
            #            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            
            # cv2.imshow('Camera Calibration', display_frame)
            
            # key = cv2.waitKey(5000)& 0xFF
            # if key == 27:  # ESC  key
            #     break
            # elif key == 32 and ret_corners:  # SPACE key and chessboard detected
            #     # Save image
            #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #     filename = os.path.join(save_dir, f"calibration_{timestamp}_{captured_count:02d}.jpg")
            #     cv2.imwrite(filename, frame)
            #     captured_count += 1
            #     print(f"Captured image {captured_count}/{num_images}: {filename}")
        
        cam.stop()
        # cv2.destroyAllWindows()
        
        print(f"Capture complete! Saved {captured_count} images to {self.image_dir}")
        return captured_count > 0
    
    def load_calibration_images(self):
        """
        Load and process calibration images from directory
        
        Args:
            image_dir: Directory containing calibration images
        """
        self.objpoints = []  # Reset object points
        self.imgpoints = []  # Reset image points
        # Get list of calibration images
        image_files = glob.glob(os.path.join(self.image_dir, "*.jpg")) + \
                     glob.glob(os.path.join(self.image_dir, "*.png"))
        
        if not image_files:
            print(f"No images found in {self.image_dir}")
            return False
        
        print(f"Processing {len(image_files)} calibration images...")
        
        valid_images = 0
        
        for image_file in image_files:
            # Read image
            img = cv2.imread(image_file)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                # Refine corner positions
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                print("Object points:", self.objp.shape, "| Image points:", corners2.shape)
                # Add object points and image points
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)
                valid_images += 1
                print(f"✓ Processed: {os.path.basename(image_file)}")
            else:
                print(f"✗ No chessboard found: {os.path.basename(image_file)}")
        
        print(f"Successfully processed {valid_images} images with detected chessboards")
        return True if valid_images > 0 else False
    
    def calibrate_camera(self, image_shape):
        """
        Perform camera calibration
        
        Args:
            image_shape: Shape of calibration images (height, width)
        """
        if len(self.objpoints) == 0 or len(self.imgpoints) == 0:
            print("Error: No valid calibration data found")
            return False
        
        print(f"Calibrating camera with {len(self.objpoints)} image pairs...")
        
        # Perform camera calibration
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, (image_shape[1], image_shape[0]), None, None
        )
        
        if ret:
            print("Camera calibration successful!")
            print(f"RMS reprojection error: {ret:.4f}")
            print(f"Camera matrix:\n{self.camera_matrix}")
            print(f"Distortion coefficients:\n{self.dist_coeffs.flatten()}")
            return True
        else:
            print("Camera calibration failed!")
            return False
    
    def save_calibration(self):
        """
        Save calibration results to file
        
        Args:
            filename: Output filename for calibration data
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("Error: No calibration data to save")
            return False
        
        calibration_data = {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'rvecs': self.rvecs,
            'tvecs': self.tvecs,
            'chessboard_size': self.chessboard_size,
            'square_size': self.square_size
        }
        if not os.path.exists(os.path.dirname(self.calibration_file)):
            os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        with open(self.calibration_file, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print(f"Calibration data saved to {self.calibration_file}")
        return True
    
    def load_calibration(self, filename=None):
        """
        Load calibration results from file
        
        Args:
            filename: Input filename for calibration data
        """
        try:
            if filename is not None:
                calibration_file = filename
            else:
                calibration_file = self.calibration_file
            with open(calibration_file, 'rb') as f:
                calibration_data = pickle.load(f)
            
            self.camera_matrix = calibration_data['camera_matrix']
            self.dist_coeffs = calibration_data['dist_coeffs']
            self.rvecs = calibration_data.get('rvecs')
            self.tvecs = calibration_data.get('tvecs')
            self.chessboard_size = calibration_data.get('chessboard_size', self.chessboard_size)
            self.square_size = calibration_data.get('square_size', self.square_size)
            
            print(f"Calibration data loaded from {self.calibration_file}")
            return True
        except FileNotFoundError:
            print(f"Calibration file {self.calibration_file} not found")
            return False
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def test_calibration(self, num_images=1):
        """
        Test the calibration by showing undistorted camera feed
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("Error: No calibration data available")
            return
        
        cam = picamera2.Picamera2(camera_num=0)
        cam.start()
        if not cam.started:
            print("Error: Could not open camera")
            return
        
        print("Testing calibration - Press ESC to exit")
        
        images_taken = 0
        while True:
            frame = cam.capture_array()
            
            # Undistort the image
            undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # Show original and undistorted side by side
            combined = np.hstack((frame, undistorted))
            combined = cv2.resize(combined, (1200, 400))
            
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Undistorted", (610, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imwrite(os.path.join(self.test_image_dir,f'Calibration Test {images_taken}.jpg'), combined)
            images_taken += 1

            if images_taken == num_images:
                print(f"Captured {num_images} test images. Exiting...")
                return -1
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        
        cam.stop()

    def calibration_managment(self):
        """
        Main calibration workflow
        """
        print("Camera Calibration")
        print("=" * 50)
        
        # Initialize calibrator
        # Standard chessboard: 9x6 internal corners
        
        while True:
            print("\nOptions:")
            print("1. Capture calibration images")
            print("2. Calibrate from existing images")
            print("3. Load existing calibration")
            print("4. Test calibration")
            print("5. Delete all calibration images")
            print("6. Exit")

            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                # Capture new calibration images
                num_images = int(input("Number of calibration images to capture (default: 20): ") or "20")
                if self.capture_calibration_images(num_images):
                    # Process the captured images
                    if self.load_calibration_images():
                        # Get image shape from first image
                        test_img = cv2.imread("calibration_images/calibration_20241210_164329_00.jpg")
                        if test_img is not None:
                            if self.calibrate_camera(test_img.shape[:2]):
                                self.save_calibration()
            
            elif choice == '2':
                # Calibrate from existing images
                image_dir = input("Enter calibration images directory (default: calibration_images): ").strip() or None
                image_dir = self.image_dir if image_dir is None else image_dir
                cal_true = self.load_calibration_images()
                if cal_true:
                    # Get image shape from first available image
                    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
                    if image_files:
                        test_img = cv2.imread(image_files[0])
                        if test_img is not None:
                            print("imagepoints:", len(self.imgpoints))
                            print(self.imgpoints[0].shape)
                            print("objpoints:", len(self.objpoints))
                            print(self.objpoints[0].shape)
                            if len(self.objpoints) != len(self.imgpoints):
                                raise ValueError(f"Mismatch: {len(self.objpoints)} object points vs {len(self.imgpoints)} image points")
                            if self.objpoints[0].shape[0] != self.imgpoints[0].shape[0]:
                                raise ValueError(f"Mismatch: Object corners {self.objpoints[0].shape[0]}  vs actual corners {self.imgpoints[0].shape[0]}")
                            if self.objpoints[0].shape[0] != self.chessboard_size[0] * self.chessboard_size[1]:
                                raise ValueError(f"Mismatch: Object corners {self.objpoints[0].shape[0]} vs expected {self.chessboard_size[0] * self.chessboard_size[1]}")
                            if self.calibrate_camera(test_img.shape[:2]):
                                print("Calibration successful")
                                self.save_calibration()
                                print("Calibration saved successfully")
            
            elif choice == '3':
                # Load existing calibration
                filename = input("Enter calibration file (default: camera_calibration.pkl): ").strip() or None
                filename = self.calibration_file if filename is None else filename
                self.load_calibration(filename)
            
            elif choice == '4':
                # Test calibration
                self.test_calibration()

            elif choice == '5':
                # delete all calibration images
                print("Deleting all calibration images...")
                for file in glob.glob(os.path.join(self.image_dir, "*")):
                    os.remove(file)
            
            elif choice == '6':
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please try again.")

class CalibrationLoader():
    def __init__(self, calibration_file="src/campi/camera_calibration.pkl"):
        """
        Initialize the calibration loader
        
        Args:
            calibration_file: Path to the calibration file
        """
        self.calibration_file = calibration_file
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None

    def load_calibration(self):
        """
        Load calibration results from file
        
        Args:
            filename: Input filename for calibration data
        """
        try:
            with open(self.calibration_file, 'rb') as f:
                calibration_data = pickle.load(f)
            
            self.camera_matrix = calibration_data['camera_matrix']
            self.dist_coeffs = calibration_data['dist_coeffs']
            self.rvecs = calibration_data.get('rvecs')
            self.tvecs = calibration_data.get('tvecs')
            
            print(f"Calibration data loaded from {self.calibration_file}")
            return True
        except FileNotFoundError:
            print(f"Calibration file {self.calibration_file} not found")
            return False
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False

