## Udacity SDC (Self-Driving Car): Advanced Lane Finding

### Introduction ###

In this project, the objetive is to create a pipeline, using advanced computer vision techniques, to find lane lines in the road in a video. The result expected should be like the following image:

![Lanes Image](./examples/example_output.jpg)

To get the above results, we need to take present the following criteria:

- Calibrate the camera: to compute the camera matrix and distrtion coefficients.
- Distortion Correction Image: usign the parameters obtained on Calibration Camera.
- Identify Color Transforms: To obtain the lane pixels from images.
- Make a Perspective Transform: Rectify each image to a "bird eye view".
- Identify left and right lane lines and fit with a curved functional form.
- Calculate the radious of curvature.
- Apply line finding on video: based on result from images, implement the final pipeline in the video.

## Developing the pipeline ###
Before starting the algorithm to identify the line lanes on the images or video, we need to consider that the elements captured from camera needs a correction due a distortion caused by the camera lens. This require the camera calibration

### Camera Calibration ###
The objective of this step is to prepare the variables required to get the camera matrix using the following functions from opencv (check [pipeline](Pipeline.ipynb)):

``` python3
## nx = 9, ny = 6
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

## ... calculate and save objpoints and imgpoints, using glob
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('camera_cal/calibration*.jpg')
n_it = len(images)

for i in range(n_it):
    img = cv2.imread(images[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    
    if ret == True:
        # append objp and corners
        objpoints.append(objp)
        imgpoints.append(corners)
```

The last script, is a lite version of the original one, that will be used to get the camera matrix using the following method from opencv:

```python3
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
```

## Distortion Correction Image ##

In the [pipeline](Pipeline.ipynb), we use all the images stored in [camera_cal](./camera_cal) to get all ```object_points``` and ```image_points```, after obtain this information we use the following method to unidistor image, based on the camera matrix:

```python
def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst, mtx, dist
```

The following image is used to do the distortion correction:
![](camera_cal/calibration3.jpg)

We obtain the following result:
![](./results/01_undisorted.jpg)

## Identify Color Transforms ##
Lets print all the rgb channels for a image taken in the lane lines context.

The following image will be used to identify the color transforms:
![](straight_lines1.jpg)

The RGB channels printed as gray scale looks like this:

- Red channel:
![](results/03_lane_lines_red.jpg)

- Green channel:
![](results/03_lane_lines_green.jpg)

- Blue channel:
![](results/03_lane_lines_blue.jpg)

Lets move to another color space.

In the HLS space, each channel printed as gray scale looks like this:
- H channel:
![](results/03_lane_lines_h_hls.jpg)

- L channel:
![](results/03_lane_lines_l_hls.jpg)

- S channel:
![](results/03_lane_lines_s_hls.jpg)

In the HSV space, each channel printed as gray scale looks like this:
- H channel:
![](results/03_lane_lines_h_hsv.jpg)

- S channel:
![](results/03_lane_lines_s_hsv.jpg)

- V channel:
![](results/03_lane_lines_v_hsv.jpg)

Based in the results, the S channel from HLS should be a good option, because it have a high contrast with the lines and background.

Applying sobel operation in x direction to the selected channel with the following method:
```
def sobel_x(img):
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    return scaled_sobel
```
The result is:
![](results/04_sobel_x.jpg)

the last method was modified, to return a binary image:
```
def sobel_x_binary(img, thresh_min, thresh_max):
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary
```

with the following parameters:

```
thresh_min = 20
thresh_max = 200
```

The result obtained is:

![](results/04_sobel_binary_x.jpg)

Another analisys was to implement a threshold in the same selected channel, with the following method:
```
def binary(img, thresh_min, thresh_max):
    binary = np.zeros_like(scaled_sobel)
    binary[(img >= thresh_min) & (img <= thresh_max)] = 1
    return binary
```

And its implementation:
```
b_image = binary(selected_channel, 120, 255)
```

The result is:
![](results/04_binary_threshold.jpg)


# Make a Perspective Transform ##
To make a perspective transform, the points are selected similar to the region of interest made in the previous project, so we use the following function made and its implementation as a reference:

```
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

(height, width) = sxbinary.shape
## Region of interest
mid_offset = 100
left_bottom = [mid_offset, height]
right_bottom = [width, height]
apex = [((width+100)//2)-mid_offset, 460]
apex2 = [((width+100)//2)+mid_offset, 460]
corners = [left_bottom, right_bottom, apex2, apex]
area = np.array( [corners], dtype=np.int32 )
img_region = region_of_interest(sxbinary, area)
```
The following code is used to check the regions that we want to check for our perspective trasnform:

```
(height, width) = img_lane.shape[:2]
mid_offset = 105
bottom_offset = 220
x_offset = -20

left_bottom = (0 + bottom_offset + x_offset, height)
right_bottom = (width - bottom_offset + x_offset + 70, height)
apex = (((width+100)//2)-mid_offset + x_offset, 470)
apex2 = (((width+100)//2)+mid_offset + x_offset - 60, 470)
corners = [left_bottom, right_bottom, apex2, apex]

def draw_region(img, vertices):
    line_color = (255, 0, 0)
    thickness = 9
    
    image = np.copy(img)
    
    image = cv2.line(image, vertices[0], vertices[1], line_color, thickness)
    image = cv2.line(image, vertices[1], vertices[2], line_color, thickness)
    image = cv2.line(image, vertices[2], vertices[3], line_color, thickness)
    image = cv2.line(image, vertices[3], vertices[0], line_color, thickness)
    return image

rgb_lane_lines = cv2.cvtColor(img_lane, cv2.COLOR_BGR2RGB)
img_region_lines = draw_region(rgb_lane_lines, corners)
```

The result is:
![](results/05_lines_perspective.jpg)

The ```corners``` variable from the last script is used to get a perspective transform. We define the following function to get `bird_view` transformation:

```
def bird_view(img, corners):
    offset = 50 # offset for dst points
    img_size = (img.shape[1], img.shape[0])

    src_points = np.float32(corners)

    dst_points = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                 [img_size[0]-offset, img_size[1]-offset], 
                                 [offset, img_size[1]-offset]])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, M_inv

bird_view, M, M_inv = bird_view(img_region, corners[::-1])
```

The result after implement ```bird_view``` function on the binary image is:
![](results/05_bird_view.jpg)

## Identify left and right lane lines ##
At htis point the perspective and binary transformations shows a nice job.
Using the half image from bird view, with the following code, the peaks asociated with the left and right lane are visible in the histogram:

```
def _lr_peaks_histogram(img):
    bottom_half = img[img.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis=0)
    return histogram, bottom_half

histogram, bottom_half = _lr_peaks_histogram(bird_view)
```

![](results/06_histogram.png)

<!-- Using sliding windows -->

## Calculate the radious of curvature



## Final pipeline ##
The final pipeline is created in [`final_implementation.py`](./final_implementation.py) file. There, each step was wrapped in a function inside a class named LanesProcessing and tested with the [`final_implementation.ipynb`](./final_implementation.ipynb) notebook.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

