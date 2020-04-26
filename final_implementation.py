import numpy as np
import os
import cv2
import glob

from moviepy.editor import VideoFileClip
from IPython.display import HTML

left_line = None
right_line = None


class LanesProcessing:
    def __init__(self, img_location):
        self.left_line = None
        self.right_line = None

        mtx, dist = calibration(img_location)

        self.mtx = None
        self.dist = None


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
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # counter to reset after 5 iterations if issues arise
        self.counter = 0

    ''' 
    Resets the line class upon failing five times in a row.
    '''

    def count_check(self):
        # self.counter += 1
        # Reset if failed five times
        if self.counter >= 5:
            self.reset()

    '''
    Fit a second order polynomial to the line.
    '''

    def fit_line(self, x_points, y_points, first_try=True):
        try:
            n = 5
            self.current_fit = np.polyfit(y_points, x_points, 2)
            self.all_x = x_points
            self.all_y = y_points
            self.recent_fit.append(self.current_fit)
            if len(self.recent_fit) > 1:
                self.diffs = (
                    self.recent_fit[-2] - self.recent_fit[-1]) / self.recent_fit[-2]
            self.recent_fit = self.recent_fit[-n:]
            self.best_fit = np.mean(self.recent_fit, axis=0)
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
calibration
this function is used to get a calibration on camera
'''


def calibration(img_location):
    nx = 9
    ny = 6

    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(img_location)
    n_it = len(images)
    for i in range(n_it):
        img = cv2.imread(images[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            # append objp and corners
            objpoints.append(objp)
            imgpoints.append(corners)

            # draw corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       gray.shape[::-1], None,
                                                       None)
    return mtx, dist


'''
sobel_x 
this function requires as input a image in grayscale
'''


def sobel_x(img, sobel_kernel=3):
    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    return scaled_sobel


'''
binary_image
this function requires as input a image in grayscale
'''


def binary(img, thresh_min, thresh_max):
    binary = np.zeros_like(img)
    binary[(img >= thresh_min) & (img <= thresh_max)] = 1
    return binary


'''
pipeline_binary
this function creates a binary image.
'''


def pipeline_binary(img, mtx, dist, s_thresh=(120, 255), sx_thresh=(10, 200),
                    r_thresh=(200, 255), sobel_kernel=3):

    distorted_img = np.copy(img)
    dst = cv2.undistort(distorted_img, mtx, dist, None, mtx)
    hls = cv2.cvtColor(dst, cv2.COLOR_BGR2HLS).astype(
        np.float)  # Convert to HLS colorspace

    r_channel = dst[:, :, 0]  # R_CHANNEL FROM RGB
    s_channel = hls[:, :, 2]  # S_CHANNEL FROM HLS
    sx = sobel_x(s_channel, sobel_kernel)  # Sobel in X direction for S_CHANNEL

    r_binary = binary(r_channel, r_thresh[0], r_thresh[1])
    s_binary = binary(s_channel, s_thresh[0], s_thresh[1])
    sx_binary = binary(sx, sx_thresh[0], sx_thresh[1])

    combined_binary = np.zeros_like(r_binary)
    combined_binary[((s_binary == 1) & (sx_binary == 1))
                    | ((sx_binary == 1) & (r_binary == 1))
                    | ((s_binary == 1) & (r_binary == 1))] = 1
    return combined_binary


'''
get_corners
function used to get corners baed on image
'''


def get_corners(img):
    height, width = img.shape[:2]
    mid_offset = 95
    bottom_offset_left = 250
    bottom_offset_right = 140
    x_offset = -10

    apex_offset = 100
    y_apex = (height//2) + apex_offset

    left_bottom = (0 + bottom_offset_left + x_offset, height)
    right_bottom = (width - bottom_offset_right + x_offset + 70, height)
    apex = (((width+100)//2)-mid_offset + x_offset, y_apex)
    apex2 = (((width+100)//2)+mid_offset + x_offset - 60, y_apex)
    corners = [left_bottom, right_bottom, apex2, apex]

    return corners


'''
bird_view
function used to get a bird_view perspective
'''


def bird_view(img):
    corners = get_corners(img)
    return bird_view_corners(img, corners)


def bird_view_corners(img, corners):
    height, width = img.shape[:2]
    # corners = get_corners(img)
    src_points = np.float32(corners)

    offset = 300  # offset for dst points
    dst_points = np.float32([
                            [offset, height],
                            [width-offset, height],
                            [width-offset, 0],
                            [offset, 0]
                            ])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped, M


def draw_region(img, vertices):
    line_color = (255, 0, 0)
    thickness = 9

    image = np.copy(img)

    image = cv2.line(image, vertices[0], vertices[1], line_color, thickness)
    image = cv2.line(image, vertices[1], vertices[2], line_color, thickness)
    image = cv2.line(image, vertices[2], vertices[3], line_color, thickness)
    image = cv2.line(image, vertices[3], vertices[0], line_color, thickness)
    return image


def sliding_window(x_current, margin, minpix, nonzerox, nonzeroy,
                   win_y_low, win_y_high, window_max, counter, side):
    # Identify window boundaries
    win_x_low = x_current - margin
    win_x_high = x_current + margin
    # Identify the nonzero pixels in x and y within the window
    good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                 & (nonzerox >= win_x_low)
                 & (nonzerox < win_x_high)).nonzero()[0]
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_inds) > minpix:
        x_current = np.int(np.mean(nonzerox[good_inds]))
    if counter >= 5:
        if win_x_high > window_max or win_x_low < 0:
            if side == 'left':
                left_tracker = False
            else:
                right_tracker = False

    return good_inds, x_current


'''
first_lines
function used to find lines pixels positions
'''

def first_lines(img, mtx, dist):
    binary_img = pipeline_binary(img, mtx, dist)
    binary_warped, perspective_M = bird_view(binary_img)

    # apply histogram to the half image to get left and right peaks.
    bottom_half = binary_warped[binary_warped.shape[0]//2:, :]
    histogram = np.sum(bottom_half, axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Number of sliding windows
    nwindows = 35

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    left_tracker = True
    right_tracker = True
    counter = 0

    # Step through the windows one by one
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        window_max = binary_warped.shape[1]
        if left_tracker == True and right_tracker == True:
            good_left_inds, leftx_current = sliding_window(leftx_current, margin, minpix, nonzerox, nonzeroy,
                                                           win_y_low, win_y_high, window_max, counter, 'left')
            good_right_inds, rightx_current = sliding_window(rightx_current, margin, minpix, nonzerox, nonzeroy,
                                                             win_y_low, win_y_high, window_max, counter, 'right')
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            counter += 1
        elif left_tracker == True:
            good_left_inds, leftx_current = sliding_window(leftx_current, margin, minpix, nonzerox, nonzeroy,
                                                           win_y_low, win_y_high, window_max, counter, 'left')
            # Append these indices to the list
            left_lane_inds.append(good_left_inds)
        elif right_tracker == True:
            good_right_inds, rightx_current = sliding_window(rightx_current, margin, minpix, nonzerox, nonzeroy,
                                                             win_y_low, win_y_high, window_max, counter, 'right')
            # Append these indices to the list
            right_lane_inds.append(good_right_inds)
        else:
            break

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # return histogram, left and right line pixel positions
    return histogram, leftx, lefty, rightx, righty


def second_ord_poly(line, val):
    a = line[0]
    b = line[1]
    c = line[2]
    return (a*val**2)+(b*val)+c


def draw_lines(img, mtx, dist):
    binary = pipeline_binary(img, mtx, dist)
    binary_warped, perspective_M = bird_view(binary)

    # Check if lines were last detected; if not, re-run first_lines
    if left_line.detected == False or right_line.detected == False:
        histogram, leftx, lefty, rightx, righty = first_lines(img, mtx, dist)
        left_line.fit_line(leftx, lefty, True)
        right_line.fit_line(rightx, righty, True)

    # Set the fit as the current fit for now
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit

    # Again, find the lane indicators
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin))
                      & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))
                       & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Set the x and y values of points on each line
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each again.
    left_fit = left_line.fit_line(leftx, lefty, False)
    right_fit = right_line.fit_line(rightx, righty, False)

    # Generate x and y values for plotting
    fity = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
    fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    left_line_window1 = np.array(
        [np.transpose(np.vstack([fit_leftx-margin, fity]))])
    left_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([fit_leftx+margin, fity])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([fit_rightx-margin, fity]))])
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([fit_rightx+margin, fity])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Calculate the pixel curve radius
    y_eval = np.max(fity)
    left_curverad = (
        (1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = (
        (1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(left_line.all_y*ym_per_pix,
                             left_line.all_x*xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_line.all_y*ym_per_pix,
                              right_line.all_x*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                           left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
                            right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    avg_rad = round(np.mean([left_curverad, right_curverad]), 0)
    rad_text = 'Radius of Curvature = {}(m)'.format(avg_rad)

    # Calculating middle of the image, aka where the car camera is
    middle_of_image = img.shape[1] / 2
    car_position = middle_of_image * xm_per_pix

    # Calculating middle of the lane
    left_line_base = second_ord_poly(left_fit_cr, img.shape[0] * ym_per_pix)
    right_line_base = second_ord_poly(right_fit_cr, img.shape[0] * ym_per_pix)
    lane_mid = (left_line_base+right_line_base)/2

    # Calculate distance from center and list differently based on left or right
    dist_from_center = lane_mid - car_position
    if dist_from_center >= 0:
        center_text = '{} meters left of center'.format(
            round(dist_from_center, 2))
    else:
        center_text = '{} meters right of center'.format(
            round(-dist_from_center, 2))

    # List car's position in relation to middle on the image and radius of curvature
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, center_text, (10, 50), font, 1, (255, 255, 255), 2)
    cv2.putText(img, rad_text, (10, 100), font, 1, (255, 255, 255), 2)

    # Invert the transform matrix from birds_eye
    Minv = np.linalg.inv(perspective_M)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result
    # return binary_warped


def process_image(img, mtx, dist):
    result = draw_lines(img, mtx, dist)

    # arr = np.zeros((result.shape[0],result.shape[1],3))
    # arr[:,:,0] = result
    # arr[:,:,1] = result
    # arr[:,:,2] = result
    # new_arr = arr.dot(255)

    return result
    # return new_arr


def main():
    ## global object to manage left and right lines
    global left_line, right_line

    left_line = Line()
    right_line = Line()

    # Location of calibration images
    img_location = 'camera_cal/calibration*.jpg'
    LanesProcessing('camera_cal/calibration*.jpg')

    # Calibrate camera and return calibration data
    mtx, dist = calibration(img_location)

    # Convert to video
    vid_output = './videos/output_vid.mp4'

    # # The file referenced in clip1 is the original video before anything has
    # clip1 = VideoFileClip('./videos/project_video.mp4').subclip(20,30)
    clip1 = VideoFileClip('./videos/challenge_video.mp4')

    # This function expects 3-channel images
    vid_clip = clip1.fl_image(lambda image: process_image(image, mtx, dist))
    vid_clip.write_videofile(vid_output, audio=False)


if __name__ == '__main__':
    main()
