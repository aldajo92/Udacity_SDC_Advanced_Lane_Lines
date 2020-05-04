import numpy as np
import os
import cv2
import glob

from moviepy.editor import VideoFileClip
from IPython.display import HTML


class LanesProcessing():
    def __init__(self, img_location, nx=9, ny=6):
        self.left_line = Line()
        self.right_line = Line()

        self.nx = nx
        self.ny = ny
        self.mtx = None
        self.dist = None
        self.m_bird_view = np.empty(1)
        self.m_inv_bird_view = np.empty(1)

        self._calibration(img_location)

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700  # meters per pixel in x dimension

    # calibration
    def _calibration(self, img_location):
        nx = self.nx
        ny = self.ny

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

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist

    # Just for proof of concept
    def bird_view_binary(self, img):
        binary = self._pipeline_binary(img)
        bird_view, _ = self._bird_view(binary)

        arr = np.zeros((bird_view.shape[0], bird_view.shape[1], 3))
        arr[:, :, 0] = bird_view
        arr[:, :, 1] = bird_view
        arr[:, :, 2] = bird_view
        new_arr = arr.dot(255)
        return new_arr

    # Sobel X operation
    def _sobel_x(self, img, sobel_kernel=3):
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        return scaled_sobel

    # Binary operation over a  1-channel image
    def _binary(self, img, thresh_min, thresh_max):
        binary = np.zeros_like(img)
        binary[(img >= thresh_min) & (img <= thresh_max)] = 1
        return binary

    # Pipeline used to select red, s, and sx channel to activate binary operation
    def _pipeline_binary(self, img, s_thresh=(120, 255), sx_thresh=(10, 200),
                         r_thresh=(200, 255), sobel_kernel=3):

        distorted_img = np.copy(img)
        dst = cv2.undistort(distorted_img, self.mtx, self.dist, None, self.mtx)

        hls = cv2.cvtColor(dst, cv2.COLOR_BGR2HLS).astype(np.float)

        r_channel = dst[:, :, 0]  # R_CHANNEL FROM RGB
        s_channel = hls[:, :, 2]  # S_CHANNEL FROM HLS
        sx = self._sobel_x(s_channel, sobel_kernel)

        r_binary = self._binary(r_channel, r_thresh[0], r_thresh[1])
        s_binary = self._binary(s_channel, s_thresh[0], s_thresh[1])
        sx_binary = self._binary(sx, sx_thresh[0], sx_thresh[1])

        combined_binary = np.zeros_like(r_binary)
        combined_binary[((s_binary == 1) & (sx_binary == 1))
                        | ((sx_binary == 1) & (r_binary == 1))
                        | ((s_binary == 1) & (r_binary == 1))] = 1
        return combined_binary

    # Get corners based on image
    def _get_corners(self, img):
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

    # Bird View operation with default corner function
    def _bird_view(self, img):
        corners = self._get_corners(img)
        return self._bird_view_corners(img, corners)

    # Bird View operation with corners parameter
    def _bird_view_corners(self, img, corners):
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

        # if (self.m_bird_view == np.empty(1))[0]:
        m_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        # Transform matrix from birds_eye
        self.m_bird_view = m_matrix
        # Invert the transform matrix from birds_eye
        self.m_inv_bird_view = np.linalg.inv(m_matrix)

        warped = cv2.warpPerspective(img, self.m_bird_view, (width, height))
        return warped

    # Helper function to draw region
    def _draw_region(self, img, vertices):
        line_color = (255, 0, 0)
        thickness = 9
        image = np.copy(img)
        image = cv2.line(image, vertices[0],
                         vertices[1], line_color, thickness)
        image = cv2.line(image, vertices[1],
                         vertices[2], line_color, thickness)
        image = cv2.line(image, vertices[2],
                         vertices[3], line_color, thickness)
        image = cv2.line(image, vertices[3],
                         vertices[0], line_color, thickness)
        return image

    # Histogram to the half image to get left and right peaks.
    def _lr_peaks_histogram(self, bird_view):
        bottom_half = bird_view[bird_view.shape[0]//2:, :]
        histogram = np.sum(bottom_half, axis=0)
        return histogram, bottom_half

    # Extract First Lines information
    def _first_lines(self, binary_warped, draw_windows=False, update_fit=True):
        histogram, bottom_half = self._lr_peaks_histogram(binary_warped)

        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        out_img = None
        if draw_windows:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Width of the windows +/- margin
        margin = 100

        # Minimum number of pixels found to recenter window
        minpix = 50

        # Empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Number of sliding windows
        nwindows = 35

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)

        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            if draw_windows:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 4)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 4)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if update_fit:
            self.left_line.fit_line(leftx, lefty, True)
            self.right_line.fit_line(rightx, righty, True)

        if draw_windows:
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

        return out_img

    # Second_order Poly to calculate the middle of the lane
    def _second_ord_poly(self, poly_values, val):
        a = poly_values[0]
        b = poly_values[1]
        c = poly_values[2]
        return (a*val**2)+(b*val)+c

    def _show_image_information(self, image, radious, dist_from_center):
        if dist_from_center >= 0:
            center_text = 'Vehicle is {} meters left of center'.format(
                round(dist_from_center, 2))
        else:
            center_text = 'Vehicle is {} meters right of center'.format(
                round(-dist_from_center, 2))

        # Show information over the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, radious, (50, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(image, center_text, (50, 100), font, 1, (255, 255, 255), 2)

    # detect left and right lanes from img and binary images precalculated
    def _detect_left_right_lanes(self, binary_warped):
        # Check if lines were last detected; if not, re-run first_lines
        if self.left_line.detected == False or self.right_line.detected == False:
            _ = self._first_lines(binary_warped)

        # Set the fit as the current fit for now
        left_fit = self.left_line.current_fit
        right_fit = self.right_line.current_fit

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 100
        l_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin))
                       & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        r_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))
                       & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Extract left and right line pixel positions
        leftx = nonzerox[l_lane_inds]
        lefty = nonzeroy[l_lane_inds]
        rightx = nonzerox[r_lane_inds]
        righty = nonzeroy[r_lane_inds]

        # Fit new polynomials
        left_fit = self.left_line.fit_line(leftx, lefty, False)
        right_fit = self.right_line.fit_line(rightx, righty, False)

        # Generate x and y values for plotting
        fity = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
        fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[nonzeroy[l_lane_inds], nonzerox[l_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[r_lane_inds], nonzerox[r_lane_inds]] = [0, 0, 255]

        return out_img, left_fit, right_fit, fity, fit_leftx, fit_rightx

    # Calculates the radius of curvature
    def _measure_curvature(self, fity, left_fit_cr, right_fit_cr):
        ym_per_pix = self.ym_per_pix  # meters per pixel in y dimension
        xm_per_pix = self.xm_per_pix  # meters per pixel in x dimension

        y_eval = np.max(fity)

        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                               left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
                                right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        avg_rad = round(np.mean([left_curverad, right_curverad]), 0)
        return avg_rad

    # process image function that returns information about finding lanes
    def process_image(self, img):
        binary = self._pipeline_binary(img)
        binary_warped = self._bird_view(binary)

        (out_img,
         left_fit,
         right_fit,
         fity,
         fit_leftx,
         fit_rightx) = self._detect_left_right_lanes(binary_warped)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.left_line.all_y*self.ym_per_pix,
                                 self.left_line.all_x*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.right_line.all_y*self.ym_per_pix,
                                  self.right_line.all_x*self.xm_per_pix, 2)

        # Calculate the pixel curve radius
        avg_rad = self._measure_curvature(fity, left_fit_cr, right_fit_cr)

        rad_text = 'Radius of Curvature = {}(m)'.format(avg_rad)

        # Calculating middle of the image, aka where the car camera is
        middle_of_image = img.shape[1] / 2
        car_position = middle_of_image * self.xm_per_pix

        # Calculating middle of the lane
        left_line_base = self._second_ord_poly(
            left_fit_cr, img.shape[0] * self.ym_per_pix)
        right_line_base = self._second_ord_poly(
            right_fit_cr, img.shape[0] * self.ym_per_pix)
        lane_mid = (left_line_base+right_line_base)/2

        # Calculate distance from center and list differently based on left or right
        dist_from_center = lane_mid - car_position

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
        pts = np.hstack((pts_left, pts_right))

        # 4. Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        img_colored_lines = cv2.addWeighted(out_img, 1, color_warp, 0.4, 0)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(
            img_colored_lines, self.m_inv_bird_view, (img.shape[1], img.shape[0]))

        # Combine the warp and original image
        result = cv2.addWeighted(newwarp, 0.4, img, 0.6, 0)

        self._show_image_information(result, rad_text, dist_from_center)
        return result


class Line():
    def __init__(self):
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # All others not carried over between first detections
        self.reset()

    def reset(self):
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients
        self.recent_fit = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

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

            return line_fit

        except (TypeError, np.linalg.LinAlgError):
            line_fit = self.best_fit
            if first_try == True:
                self.reset()

            return line_fit


# Function to get input video and transform using LanesProcessing Object
def main():
    # Lane processing object
    processing = LanesProcessing('camera_cal/calibration*.jpg')

    # Path to the clip input
    video_input = './videos/project_video.mp4'
    # Path to the clip output
    video_output = './videos/output_vid.mp4'

    clip1 = VideoFileClip(video_input)

    # This operation expects 3-channel images
    vid_clip = clip1.fl_image(lambda frame: processing.process_image(frame))
    vid_clip.write_videofile(video_output, audio=False)

# Just for proof of concepts


def generate_bird_view_video():
    # Lane processing object
    processing = LanesProcessing('camera_cal/calibration*.jpg')

    # Path to the clip input
    video_input = './videos/project_video.mp4'
    # Path to the clip output
    video_output = './videos/bird_view_binary.mp4'

    clip1 = VideoFileClip(video_input).subclip(20, 30)

    # This operation expects 3-channel images
    vid_clip = clip1.fl_image(lambda frame: processing.bird_view_binary(frame))
    vid_clip.write_videofile(video_output, audio=False)


if __name__ == '__main__':
    main()
