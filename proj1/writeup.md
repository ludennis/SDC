# **Proj 1: Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### 1. The Pipeline ###

My pipeline consisted of the following steps:
1. converting the image to grayscale
2. applying gaussian blur to the image
3. finding canny edges within the blurred image
4. finding lines using hough lines 
5. connecting all the lines with average slope and average intercept

### 2. The "extrapolate_lines()" function (my own "draw_lines()" function) ###

The extrapolate_lines() function takes in the following as inputs:
1. image
2. lines found through using hough lines
3. vertices 

What the function extrapolate_lines() do:
1. finds the slope of each line
2. determines whether the line is on the left or on the right
3. finds the average slope and intercept of all the lines on either side
4. extrapolates a single line on both left and right side


### 3. Shortcomings ###
1. extrapolated lines are somewhat noisy 
2. extrapolated lines are not accurate in the challenge video


### 4. Possible Improvements ### 
1. To use a filter for extrapolating lines
