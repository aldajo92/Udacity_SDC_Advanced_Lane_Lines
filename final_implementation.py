import numpy as np
import os
import cv2
import matplotlib.image as mpimg
import glob

# Import the libraries required for video
from moviepy.editor import VideoFileClip
from IPython.display import HTML


class Line():
    def __init__(self):
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # Initialize all others not carried over between first detections
        self.reset()

    def reset(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # recent polynomial coefficients
        self.recent_fit = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # x values for detected line pixels
        self.allx = None  
        # y values for detected line pixels
        self.ally = None
        # counter to reset after 5 iterations if issues arise
        self.counter = 0        

    def count_check(self):
        ''' 
        Resets the line class upon failing five times in a row.
        '''
        # Increment the counter - NOT IMPLEMENTED
        #self.counter += 1
        # Reset if failed five times
        if self.counter >= 5:
            self.reset()

    def fit_line(self, x_points, y_points, first_try=True):
        '''
        Fit a second order polynomial to the line.
        The challenge videos sometimes throws errors, so the below trys first.
        Upon the error being thrown, either reset the line or add to counter.
        '''
        try: 
            n = 5
            self.current_fit = np.polyfit(y_points, x_points, 2)
            self.all_x = x_points
            self.all_y = y_points
            self.recent_fit.append(self.current_fit)
            if len(self.recent_fit) > 1:
                self.diffs = (self.recent_fit[-2] - self.recent_fit[-1]) / self.recent_fit[-2]
            self.recent_fit = self.recent_fit[-n:]
            self.best_fit = np.mean(self.recent_fit, axis = 0)
            line_fit = self.current_fit
            self.detected = True
            self.counter = 0

            return line_fit

        except (TypeError, np.linalg.LinAlgError):
            line_fit = self.best_fit
            if first_try == True:
                self.reset()
            else:
                self.count_check()

            return line_fit


'''
Camera calibration.
'''
def calibration(images, nx, ny):
    # Load in the chessboard calibration images to a list
    c_images = []

    for fname in cal_image_loc:
        img = mpimg.imread(fname)
        c_images.append(img)

    # Prepare object points
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays for later storing object points and image points
    objpoints = []
    imgpoints = []

    # Iterate through images for their points
    n_it = len(images)
    for i in range(n_it):
        img = cv2.imread(images[i])
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    # Returns camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                       gray.shape[::-1], None, 
                                                       None)

    return mtx, dist

def pipeline_binary(img, mtx, dist, s_thresh=(125, 255), sx_thresh=(10, 100), 
             R_thresh = (200, 255), sobel_kernel = 3):
    ''' 
    Pipeline to create binary image.
    This version uses thresholds on the R & S color channels and Sobelx.
    Binary activation occurs where any two of the three are activated.
    '''
    distorted_img = np.copy(img)
    dst = cv2.undistort(distorted_img, mtx, dist, None, mtx)
    # Pull R
    R = dst[:,:,0]
    
    # Convert to HLS colorspace
    hls = cv2.cvtColor(dst, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobelx - takes the derivate in x, absolute value, then rescale
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) 
             & (scaled_sobelx <= sx_thresh[1])] = 1

    # Threshold R color channel
    R_binary = np.zeros_like(R)
    R_binary[(R >= R_thresh[0]) & (R <= R_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # If two of the three are activated, activate in the binary image
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) & (sxbinary == 1)) 
                     | ((sxbinary == 1) & (R_binary == 1))
                     | ((s_binary == 1) & (R_binary == 1))] = 1

    return combined_binary


'''
Birds eye first undistorts the image
'''
def birds_eye(img, mtx, dist):
    # Put the image through the pipeline to get the binary image
    binary_img = pipeline(img, mtx, dist)

    # Grab the image shape
    img_size = (binary_img.shape[1], binary_img.shape[0])

    # Source points - defined area of lane line edges
    src = np.float32([[690,450],[1110,img_size[1]],[175,img_size[1]],[595,450]])

    # 4 destination points to transfer
    offset = 300 # offset for dst points
    dst = np.float32([[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]],[offset, 0]])
    
    # Use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Use cv2.warpPerspective() to warp the image to a top-down view
    top_down = cv2.warpPerspective(binary_img, M, img_size)

    return top_down, M

def main():
    # Grab location of calibration images
    cal_image_loc = glob.glob('camera_cal/*.jpg')

    # Calibrate camera and return calibration data
    mtx, dist = calibration(cal_image_loc, 9, 6)

    # Convert to video
    # vid_output is where the image will be saved to
    vid_output = 'reg_vid.mp4'

    # The file referenced in clip1 is the original video before anything has 
    #   been done to it
    clip1 = VideoFileClip('project_video.mp4')

    # NOTE: this function expects color images
    vid_clip = clip1.fl_image(lambda image: process_image(image, mtx, dist))
    vid_clip.write_videofile(vid_output, audio=False)

if __name__ == '__main__':
    # Set the class lines equal to the variables used above
    left_line = Line()
    right_line = Line()
    main()