# Canny edge detector from scratch
#### Video Demo: https://youtu.be/u9NkQgAeQbc

### Packages Used:
#### argparse, numpy, PIL

## Terminal code to execute program
### python project.py imagefile

## Description:
### Canny edge detector is a algorithm that depictis edges trhough highlighting high contrast of an Grayscale image
### Following the formula that takes in the three color cahnnel RGB: 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
### Reformatted image using larger dimension for 1:1 aspect ratio to avoid errors when calacuting grayscale image.
### redistruibted values based on largest value of matrix

## Gaussian filter
### Selection of a odd sized filter to be convolved with the image reduces noise of the image
### padding is a process that adds additional layers to the matrix based on programmers control. Padding allows for convolution to occur without going out of bounds
### When (convolving) multiplying the 5 x 5 filter by the same matrix of the image result in an average value. Reducing effects of outliers on pixel values
### Padding of size 2 around the image to apply 5 by 5 filter

## Finding the intensity gradient of the image
### Using a filter over the image such as sobel, prewitt, and roberts to mutiply with the image to fidn the gradient magnitude and direction
### I used the sobel filter due to inexpensive comutationaly


## Gardient magnitude thresholding or lower bound cut-off supression
### Selecting neighbors location to compare the pixel values based on angle value of current pixel resulting in current pixel value to zero if not larger than neighbors.
### This process trims the low value edges.

## Double Threshold
### Selecting values for low and high filters values between 0 to 1 multiplied by the max value in the matrix.
### values less than low filter become 0, values between low and high filters become low, and the no change for values eqaul or greater than high filter

## Edge Tracking by hysteresis
### Accoutning for the 8 nearest neightbor to the current pixel to determine if a pixel is above the high filter value. if a high value is within the vacinity the current value is kept other wise replaced by 0




## Implemented [Document](https://en.wikipedia.org/wiki/Canny_edge_detector#Walkthrough_of_the_algorithm) followed
