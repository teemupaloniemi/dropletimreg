

#=========================imports START HERE=========================================================================


import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd
import os
from tabulate import tabulate
from PIL import Image
from PIL.ExifTags import TAGS

# END imports


#=========================create_parabolic_kernel STARTS HERE=========================================================================


# this is a utility function for process_image function
def create_parabolic_kernel(size=12, factor=3, upside_down=False):
    """Creates a parabolic kernel of given size."""
    y = np.linspace(-1, 1, size)
    x = (y**2) * factor  # Adjusted Parabolic equation
    
    # Normalize to [0, 1]
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    
    threshold = 0.5  # Adjusted threshold for binarization
    kernel = np.outer(x, np.ones(size))
    kernel = (kernel > threshold).astype(np.uint8)  # Binarize
    
    if upside_down:
        kernel = np.flipud(kernel)  # Flip the matrix vertically
    print(kernel)
    return kernel
    #END create_parapolic_kernel


#=========================process_image STARTS HERE=========================================================================


def process_image(image, name):

    # threshold1: This is the lower threshold for the hysteresis process.
    # Any edges with an intensity gradient below this value are immediately discarded as non-edges.
    # In other words, threshold1 is used to remove noise.
    t1 = 18

    # threshold2: This is the higher threshold for the hysteresis process.
    # Any edges with an intensity gradient above this value are immediately considered as edges.
    # So, threshold2 is used to define strong edges.
    t2 = 20

    #========= important stuff starts here *** =========


    # Assuming 'image' is your input image and 't1' and 't2' are your Canny thresholds.
    edges = cv2.Canny(image, threshold1=t1, threshold2=t2)

    # Create a 3x3 matrix filled with ones. This will be used as a kernel for dilation and erosion operations.
    horizontal_kernel = np.ones((2,3), np.uint8)
    
    #if ("Top" in name): parabolic_kernel = create_parabolic_kernel(upside_down=True)
    #else: parabolic_kernel = create_parabolic_kernel() # should be false on default
    # Dilate the edges. Dilation adds pixels to the boundaries of objects in an image.
    # This can help to strengthen the detected edges.
    dilated_edges = cv2.dilate(edges, horizontal_kernel, iterations=1)

    # Erode the dilated edges. Erosion removes pixels from the boundaries of objects in an image.
    # This can help to remove noise and small unwanted details.
    eroded_edges = cv2.erode(dilated_edges, horizontal_kernel, iterations=1)

    # Find the contours in the eroded edges image. Contours are simply the boundaries of the connected objects.
    # The function returns a list of contours and a hierarchy (which is not used here, hence the underscore).
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image (all zeros) with the same shape as the eroded edges image.
    # This will be used to draw the filtered contours.
    filtered_edges = np.zeros_like(dilated_edges)

    # Loop through each detected contour.
    for contour in contours:
        # Check if the length (perimeter) of the contour is greater than a threshold (100 in this case).
        if cv2.arcLength(contour, True) > 50:
            # Draw the contour on the filtered_edges image. The contour is filled with white (255).
            cv2.drawContours(filtered_edges, [contour], -1, (255), thickness=cv2.FILLED)

    # Use HoughCircles to detect circles in the edges image.
    circles = cv2.HoughCircles(filtered_edges, cv2.HOUGH_GRADIENT, dp=1, minDist=60, param1=20, param2=10, minRadius=30, maxRadius=60)

    # Flag to track if a half-circle was found
    half_circle_found = False

    # If some circles are detected, filter and draw them.
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Check if the circle is in the upper half of the image
            if i[1] < image.shape[0] / 2:
                # Draw the outer circle
                cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
                half_circle_found = True

    # Print the result based on the flag
    if half_circle_found:
        print(f"Half-circle resembling a water droplet was successfully found in {name}.")
    else:
        print(f"No half-circle resembling a water droplet was found in {name}.")

            

    #========= important stuff ends here *** =========

    count = 0
    # Binarize the *_edges matrix
    binary_matrix = np.where(edges > 0, 1, 0)
    
    # double check if anything found
    for i in range(binary_matrix.shape[0]):
        for j in range(binary_matrix.shape[1]):
            if binary_matrix[i][j] == 1:
                count += 1

    if (count == 0):
        image_top, image_bottom = cutted_picture(2) #2 = something else than 1, True for zooming
        if ("Top" in name):
            return process_image(image_top, name)
        elif ("Bottom" in name):
            return process_image(image_bottom, name)
        return # if those dont work idk what this helps 

    # Find the first and last columns that contain a '1'
    first_column_with_one = np.where(binary_matrix.any(axis=0))[0][0]
    last_column_with_one = np.where(binary_matrix.any(axis=0))[0][-1]

    # Calculate the difference
    difference = last_column_with_one - first_column_with_one

    if (difference < 20):
        print("Less than 20, retrying")
        image_top, image_bottom = cutted_picture(2) #2 = something else than 1, True for zooming
        if ("Top" in name):
            return process_image(image_top, name)
        elif ("Bottom" in name):
            return process_image(image_bottom, name)
        return # if those dont work idk what this helps 

    # Print the first and last columns
    print(f"{name} - First column with 1: ", first_column_with_one)
    print(f"{name} - Last column with 1: ", last_column_with_one)
    print(f"{name} - Difference: ",
          last_column_with_one - first_column_with_one)

    # Check if the image is grayscale, if not convert it to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Draw vertical lines on the original image
    # Convert to BGR for colored lines
    image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.line(image_with_lines, (first_column_with_one, 0),
             (first_column_with_one, image.shape[0]), (0, 0, 255), 1)
    cv2.line(image_with_lines, (last_column_with_one, 0),
             (last_column_with_one, image.shape[0]), (0, 0, 255), 1)

    # Save the output image
    cv2.imwrite(os.path.join(
        output_dir, f'{name}_with_lines{filetype}'), image_with_lines)

    # Display the image with lines
    # Convert to RGB for correct color display in matplotlib
    plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
    plt.title(name)
    plt.show()

    # Save the output image
    cv2.imwrite(os.path.join(output_dir, f'{name}_edges{filetype}'), edges)

    # Display the output image
    #uncomment below plt.imshows to see what is happeinnng between the stars in the beginning of this function
    plt.imshow(edges, cmap='gray')
    plt.title(f'{name} Regualar edges')
    plt.show()
    plt.imshow(filtered_edges, cmap='gray')
    plt.title(f'{name} Dialated, Eroded and Filtered edges')
    plt.show()
    
    return difference
    # END process_image


#=========================cutted_picture STARTS HERE=========================================================================


def cutted_picture(a):
    # these are global variables that we use for zooming
    global orig_left_proportion, orig_top_proportion, orig_right_proportion, orig_bottom_proportion

    # Open the image
    image = Image.open(os.path.join(input_dir, bmp_file))

    # Define crop proportions. These are shown as a green rectangle.
    if a == 0 and bmp_file == bmp_files[0]:  # Only for the first image
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get ROI from the user
        x, y, w, h = get_roi(image_cv)
        
        # Convert ROI to proportions
        orig_left_proportion = x / image.width
        orig_top_proportion = y / image.height
        orig_right_proportion = (x + w) / image.width
        orig_bottom_proportion = (y + h) / image.height

    if a == 0:
        # original crop values, starting here zooming later
        left_proportion = orig_left_proportion
        top_proportion = orig_top_proportion
        right_proportion = orig_right_proportion
        bottom_proportion = orig_bottom_proportion
        print("Original corp")
    elif (orig_left_proportion - 0.05 > 0.0 
          and orig_top_proportion - 0.05 > 0.0 
          and orig_right_proportion + 0.05 < 1.0 
          and orig_bottom_proportion + 0.05 < 1.0): 
        # stop if some crop line hits a wall 

        left_proportion = orig_left_proportion-0.05  # %from the left
        orig_left_proportion = left_proportion  
        
        top_proportion = orig_top_proportion-0.05  # %from the top
        orig_top_proportion = top_proportion  
        
        right_proportion = orig_right_proportion+0.05  # %from the left
        orig_right_proportion = right_proportion 
        
        bottom_proportion = orig_bottom_proportion+0.05 # %from the top
        orig_bottom_proportion = bottom_proportion
        print("Zooming out")

    # Calculate actual pixel locations for cropping
    left = int(left_proportion * image.width)
    top = int(top_proportion * image.height)
    right = int(right_proportion * image.width)
    bottom = int(bottom_proportion * image.height)
    # Copy of the original image that we can draw a croptangle
    image_orig_crop_rect = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Draw the crop rectangle on the original image
    cv2.rectangle(image_orig_crop_rect, (left, top),
                  (right, bottom), (0, 255, 0), 2)  # 3 is the line thickness
    # Display the original image with crop rectangle
    plt.imshow(cv2.cvtColor(image_orig_crop_rect, cv2.COLOR_BGR2RGB))
    plt.title('Original Image with Crop Rectangle')
    plt.show()

    image = image.crop((left, top, right, bottom))

    # Convert the cropped PIL image back to OpenCV image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Split the image into two
    image_top = image[:image.shape[0]//2, :]
    image_bottom = image[image.shape[0]//2:image.shape[0], :]

    # Save and display the split images
    cv2.imwrite(os.path.join(output_dir, f'image_top{filetype}'), image_top)
    #plt.imshow(cv2.cvtColor(image_top, cv2.COLOR_BGR2RGB))
    #plt.show()

    cv2.imwrite(os.path.join(
        output_dir, f'image_bottom{filetype}'), image_bottom)
    #plt.imshow(cv2.cvtColor(image_bottom, cv2.COLOR_BGR2RGB))
    #plt.show()

    return image_top, image_bottom
    # END cutted_picture


#=========================get_roi STARTS HERE=========================================================================


def get_roi(image):
    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use selectROI to get the rectangle coordinates
    r = cv2.selectROI(image_rgb)
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
    return r
#END roi 


#=========================MAIN CODE STARTS HERE=========================================================================


a = 0

# original crop values, starting here zooming later
orig_left_proportion = 0.50  # %from the left
orig_top_proportion = 0.45  # %from the top
orig_right_proportion = 0.75  # %from the left
orig_bottom_proportion = 0.65  # %from the top

# 1
input_dir = "./KasperiP/KP20230628_1/"  # This is your main folder

# 2
output_dir = "outputimages"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# 3
filetype = ".bmp"

# Initialize lists to store data for the table
file_names = []
top_differences = []
bottom_differences = []
print(os.listdir(input_dir))
# Get a list of all BMP files in the current directory
bmp_files = os.listdir(input_dir)
# [f for f in os.listdir(input_dir)
#            if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(f'{filetype}')]

print("Starting to process images...")
print("Number of images to precess:", len(bmp_files))
print(bmp_files)

# Process each BMP file
for bmp_file in bmp_files:

    # Reset the values at the start of each loop
    orig_left_proportion = 0.50  # %from the left
    orig_top_proportion = 0.45  # %from the top
    orig_right_proportion = 0.75  # %from the left
    orig_bottom_proportion = 0.65  # %from the top

    image_top, image_bottom = cutted_picture(a) # a for original start, False for no zooming yet

    # Process each image and store the differences
    top_difference = process_image(image_top, f'{bmp_file} Top')
    bottom_difference = process_image(image_bottom, f'{bmp_file} Bottom')

    # Add data to the lists
    file_names.append(bmp_file)
    top_differences.append(top_difference)
    bottom_differences.append(bottom_difference)

print("Done with processing images")
# END main


#=========================table generation STARTS HERE=========================================================================


kerroin = 11.1/1032

top_differences = [
    kerroin * x if x is not None else 0 for x in top_differences]
bottom_differences = [
    kerroin * x if x is not None else 0 for x in bottom_differences]

# Create a dictionary to associate file names with data dictionaries
file_data = {}

# Populate the file_data dictionary
for file_name, d1, d2 in zip(file_names, top_differences, bottom_differences):
    file_data[file_name] = {"top_differences": d1, "bottom_differences": d2}

# Sort the file names
sorted_file_names = sorted(file_names)

# Access the sorted file names and their corresponding data
for file_name in sorted_file_names:
    data_dict = file_data[file_name]

# Create a list to store the table rows
table = []

# Populate the table rows
for file_name, d1, d2 in zip(file_names, top_differences, bottom_differences):
    table.append([file_name, d1, d2])

# Sort the table rows based on the file names
sorted_table = sorted(table, key=lambda x: x[0])

# Generate the table
headers = ["File Name", "Top Difference", "Bottom Difference"]
table_str = tabulate(sorted_table, headers, tablefmt="grid")

# Print the table
print(table_str)
#END table generation