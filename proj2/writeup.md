# Advance Lane-Finding Project
In this project, advance lane-finding through front mounted camera is to be completed through the following steps:
* camera calibration
    * compute calibration matrix and distortion coefficients given a set of chessboard images
    * apply distortion to raw iamges
* image gradient and color filter
    * sobel transform and HLS s-channel filters
    * perspective transform to birds-eye view
* lane-finding with curvature
    * detect lane pixels with sliding windows and fit to find lane boundaries
    * determine curvature of lane and position with respect to the center of vehicle
    * warp the detected lane boundaries back onto the original image
   
Written functional modules are stored in ./src folder, output images in ./output_images, and output video in ./output_video.
​
[//]: # (Image References)
​
[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
​
## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
​
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
​
---
​
### Writeup / README
​
#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  
​
You're reading it!
​
### Camera Calibration
​
#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
​
The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  
​
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
​
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
​
![alt text][image1]
​
### Pipeline (single images)
​
#### 1. Provide an example of a distortion-corrected image.
​
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
​
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
​
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)
​
![alt text][image3]
​
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
​
The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:
​
```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```
​
This resulted in the following source and destination points:
​
​
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 552, 480      | 300, 10       | 
| 252, 690      | 300, 690      |
| 1070, 690     | 900, 690      |
| 735, 480      | 900, 10       |
​
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
​
![alt text][image4]
​
#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
​
Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:
​
![alt text][image5]
​
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
​
I did this in lines # through # in my code in `my_other_file.py`
​
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
​
I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:
​
![alt text][image6]
​
---
​
### Pipeline (video)
​
#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
​
Here's a [link to my video result](./output_video/project_video.mp4)
​
---
​
### Discussion
​
#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
​
Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. This pipeline may fail when lane lines are hard to be seen. If the vehicle is in an open-road area without lane lines, it will fail. 

To make this pipeline more robust, sanity checks and smoothing techniques can be added. These techniques can help rule out outliers (noise) and smooth out the detection.
​
​