# Image Processing Script README
## Overview

This script processes a set of BMP images to detect specific features and patterns. It uses various image processing techniques, such as edge detection, dilation, erosion, and contour detection, to analyze the images. The results are then tabulated and visualized using plots.

## Dependencies
```
OpenCV (cv2)
Matplotlib (matplotlib.pyplot)
NumPy (numpy)
PIL (PIL.Image, PIL.ExifTags)
Pandas (pandas)
Tabulate (tabulate)
```

## How the Script Works

#### Imports: 

All necessary libraries are imported at the beginning.

#### Image Processing Functions:

process_image(image, name): Processes the given image to detect edges, dilate, erode, and find contours. It then calculates the difference between the first and last columns containing a '1' in the binary matrix of the image.
cutted_picture(a): Crops the image based on predefined proportions and splits it into top, bottom, and center sections.
get_roi(image): Allows the user to select a region of interest (ROI) on the image.

#### Main Code:

The script processes each BMP file in the specified directory.
For each image, it calculates the difference in the top, bottom, and center sections.
The results are stored in lists for further analysis.

#### Table Generation:

The results are tabulated using the tabulate library.
Plots are generated to visualize the differences for each image.
A moving average is applied to the data to smooth out the results, and the smoothed data is also plotted.

## Usage

Set the input_dir variable to the directory containing the BMP images you want to process.
Set the output_dir variable to the directory where you want to save the processed images.
Run the script.
For the first image, you will be prompted to select a region of interest (ROI). This will set the cropping proportions for all subsequent images.
The script will process each image, display the results, and generate plots.

## Notes

The script uses global variables for various settings, such as cropping proportions and display settings. Adjust these as needed.
The show_images variable controls whether the script displays intermediate and final results. Set it to False to disable image display.