## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Overview
Lane detection project using computer vision techniques. Much of the code is leveraged from the lecture quizs and practices. 
The following are the steps performed by the pipeline:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The output video is annotated_output.mp4, and is uploaded to [![YouTube] (https://www.youtube.com/watch?v=FGOXWpqLYi0&feature=youtu.be)]

### Requirements
* Python 3.5+
* Numpy
* OpenCV-Python
* Matplotlib
* Pickle
* Pandas

# How to run
python gen_video.py

### Camera Calibration
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the images camera_cal/calibration2.jpg and camera_cal/calibration5.jpg using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

The calibration matrix is stored as a pickle file 'calibrate_camera.p' for later usage.

### Pipeline
The following images show the two original distorted images and their corrected version.
![img1]()

#### Color Thresholding 
In project one I used yellow and white color as target colors to detect the lane, and I am using the same approach in this project since the lanes are either yellow or white. HSV and HLS are better than RGB because they separates color components from lightness and saturation. It is easier to find the thresholds on the three channels because they are now independent while in RGB space color, lightness and saturation are combines together. I compared HSV and HSL color spaces and finally decided to use HSL because HSL gives better result. 
[!img2]()

Below is the binary image output from the color thresholding.
[!img3]()

#### Gradient Thresholding
Gradient is the change of the image intensity. I used Sobel operator to find the gradients and their directions. 

There are three methods performed:

* Absolute horizontal Sobel operator on the image
* Sobel operator in both horizontal and vertical directions and calculate its magnitude
* Sobel operator to calculate the direction of the gradient

Below are the output figures after applying the three methods, and the combined binary image.
[!img4]()

#### Perspective Transform
Next step is to transform the image to a "bird's eye view" which can be used to fit a polynomial function. The very first step to accomplish this is to pick the source and destination points. Below is the table of the pixel positions of source/destination points.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 450      | 200, 0        | 
| 170, 720      | 200, 720      |
| 1140, 720     | 1000, 720      |
| 710, 450      | 1000, 0        |

Here is the two example images after applying perspective transform:
[!img5]()

#### Polynomial fit
I did the following to fit second order polynomial on both left and right lanes:
* Calculate a histogram of the bottom half of the image to identify the x positions where the pixel intensities are highest
* Partition the image into 9 horizontal slices and use slide window search to capture the coordinates of the lines
  -> Starting from the bottom slice, enclose a 200 pixel wide window around the left peak and right peak of the histogram (split the histogram in half vertically)
  -> Go up the horizontal window slices to find pixels that are likely to be part of the left and right lanes, recentering the sliding windows opportunistically
* Given 2 groups of pixels (left and right lane line candidate pixels), fit a 2nd order polynomial to each group, which represents the estimated left and right lane lines

#### Radius of Curvature
The equation of calculating the radius is [![here] (http://www.intmath.com/applications-differentiation/8-radius-curvature.php)]. We have to convert from pixel space to meters (aka real world units) by defining the appropriate pixel height to lane length and pixel width to lane width ratios:
I also converted the pixel space to meters, assuming 30 meters per 720 pixels in the vertical direction, and 3.7 meters per 700 pixels in the horizontal direction. The results are averaged.

#### Position of the Vehicle With Respect to Center
I assume the vehicle is at the middle of the two lanes. The lane's center is calculated as the mean x value of the bottom x value of the left lane line and bottom x value of the right lane line. The offset is calculated by the difference between the image center and the lane's center.


#### Annotated image
Below is the annotated image after applying previous steps:
[!img6]()

### Discussion
The pipeline implemented has made lots of strong assumptions that will be easily violated in the real world. For example, in the challenge project video, the car is on a carpool lane where you can see the carpool symbol which is not considered in this pipeline. Also the pipeline assumes there is no car in front of itself. This is almost not possible in the real world. The future work includes taking out these strong assumptions and use deep learning algorithms for better detection as deep learning has been proven very useful in computer vision.
