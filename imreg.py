

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

#=========================process_image STARTS HERE=========================================================================


def process_image(image, name):
    global show_images, STOP

    # threshold1: This is the lower threshold for the hysteresis process.
    # Any edges with an intensity gradient below this value are immediately discarded as non-edges.
    # In other words, threshold1 is used to remove noise.
    t1 = 19

    # threshold2: This is the higher threshold for the hysteresis process.
    # Any edges with an intensity gradient above this value are immediately considered as edges.
    # So, threshold2 is used to define strong edges.
    t2 = 30

    #========= important stuff starts here *** =========


    # Assuming 'image' is your input image and 't1' and 't2' are your Canny thresholds.
    edges = cv2.Canny(image, threshold1=t1, threshold2=t2)

    # Create a 3x3 matrix filled with ones. This will be used as a kernel for dilation and erosion operations.
    horizontal_kernel = np.ones((2,2), np.uint8)
    
    # Dilate the edges. Dilation adds pixels to the boundaries of objects in an image.
    dilated_edges = cv2.dilate(edges, horizontal_kernel, iterations=1)

    # Erode the dilated edges. Erosion removes pixels from the boundaries of objects in an image.
    # This can help to remove noise and small unwanted details.
    eroded_edges = cv2.erode(dilated_edges, horizontal_kernel, iterations=1)

    # Find the contours in the eroded edges image. Contours are simply the boundaries of the connected objects.
    # The function returns a list of contours and a hierarchy (which is not used here, hence the underscore).
    contours, _ = cv2.findContours(eroded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image (all zeros) with the same shape as the eroded edges image.
    # This will be used to draw the filtered contours.
    filtered_edges = np.zeros_like(dilated_edges)

    # Loop through each detected contour.
    for contour in contours:
        # Check if the length (perimeter) of the contour is greater than a threshold (100 in this case).
        if cv2.arcLength(contour, True) > 100:
            # Draw the contour on the filtered_edges image. The contour is filled with white (255).
            cv2.drawContours(filtered_edges, [contour], -1, (255), thickness=cv2.FILLED)

    if "Center" in name: 
        # Use HoughCircles to detect circles in the edges image.
        circles = cv2.HoughCircles(filtered_edges, cv2.HOUGH_GRADIENT, dp=1, minDist=90, param1=20, param2=10, minRadius=20, maxRadius=50)

        # Flag to track if a half-circle was found
        half_circle_found = False

        # Create a temporary image to draw the circles
        tmp_image = image.copy()

        # If some circles are detected, filter and draw them.
        if circles is not None:
            half_circle_found = True
            circles = np.uint16(np.around(circles))
            
            # Sort circles based on x-coordinates
            sorted_circles = sorted(circles[0, :], key=lambda x: x[0])

            # Get the leftmost and rightmost circles
            leftmost_circle = sorted_circles[0]
            rightmost_circle = sorted_circles[-1]

            for i in circles[0, :]:
                # Check if the circle is in the upper half of the image
                if i[1] < image.shape[0] / 2:
                    # Draw the outer circle on the temporary image
                    cv2.circle(tmp_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # Draw the center of the circle on the temporary image
                    cv2.circle(tmp_image, (i[0], i[1]), 2, (0, 0, 255), 3)
                    half_circle_found = True

            leftmost_point = (leftmost_circle[0] + leftmost_circle[2], leftmost_circle[1])
            rightmost_point = (rightmost_circle[0] - rightmost_circle[2], rightmost_circle[1])

            # Draw vertical lines on the edges of the leftmost and rightmost circles
            cv2.line(tmp_image, (leftmost_point[0], 0), (leftmost_point[0], image.shape[0]), (255, 0, 0), 2)
            cv2.line(tmp_image, (rightmost_point[0], 0), (rightmost_point[0], image.shape[0]), (255, 0, 0), 2)

        # Print the result based on the flag
        if half_circle_found and show_images:
            print(f"Half-circle resembling a water droplet was successfully found in {name}.")
            plt.imshow(cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB))
            plt.title(name)
            plt.show()
        elif (show_images):
            print(f"No half-circle resembling a water droplet was found in {name}.")

    #========= important stuff ends here *** =========
    
    # If we are looking for the center return center distace. 
    if "Center" in name and half_circle_found:
        if rightmost_point[0] - leftmost_point[0] > 100: return 0
        else: return rightmost_point[0] - leftmost_point[0]


    count = 0
    # Binarize the *_edges matrix
    binary_matrix = np.where(filtered_edges > 0, 1, 0)
    
    # double check if anything found
    for i in range(binary_matrix.shape[0]):
        for j in range(binary_matrix.shape[1]):
            if binary_matrix[i][j] == 1:
                count += 1

    if (count == 0):
        if STOP: 
            STOP = False
            return 0
        image_top, image_bottom, image_center = cutted_picture(2) #2 = something else than 1, True for zooming
        if ("Top" in name):
            return process_image(image_top, name)
        elif ("Bottom" in name):
            return process_image(image_bottom, name)
        return process_image(image_center, name) 

    # Find the first and last columns that contain a '1'
    first_column_with_one = np.where(binary_matrix.any(axis=0))[0][0]
    last_column_with_one = np.where(binary_matrix.any(axis=0))[0][-1]

    # Calculate the difference
    difference = last_column_with_one - first_column_with_one

    if (difference < 20):
        if STOP: 
            STOP = False
            return 0
        print("Less than 20, retrying")
        image_top, image_bottom, image_center = cutted_picture(2) #2 = something else than 1, True for zooming
        if ("Top" in name):
            return process_image(image_top, name)
        elif ("Bottom" in name):
            return process_image(image_bottom, name)
        return process_image(image_center, name) 

    # Print the first and last columns
    if (show_images):
        print(f"{name} - First column with 1: ", first_column_with_one)
        print(f"{name} - Last column with 1: ", last_column_with_one)
        print(f"{name} - Difference: ",last_column_with_one - first_column_with_one)

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
    if (show_images):    
        plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.show()

    # Save the output image
    cv2.imwrite(os.path.join(output_dir, f'{name}_edges{filetype}'), edges)

    # Display the output image
    #uncomment below plt.imshows to see what is happeinnng between the stars in the beginning of this function
    if (show_images):
        #plt.imshow(edges, cmap='gray')
        #plt.title(f'{name} Regualar edges')
        #plt.show()
        plt.imshow(filtered_edges, cmap='gray')
        plt.title(f'{name} Dialated, Eroded and Filtered edges')
        plt.show()
    
    return difference
    # END process_image


#=========================cutted_picture STARTS HERE=========================================================================


def cutted_picture(a):
    # these are global variables that we use for zooming
    global show_images, STOP, orig_left_proportion, orig_top_proportion, orig_right_proportion, orig_bottom_proportion, reset_orig_left_proportion, reset_orig_top_proportion, reset_orig_right_proportion, reset_orig_bottom_proportion

    # Open the image
    image = Image.open(os.path.join(input_dir, bmp_file))

    image_width = image.width
    image_height = image.height

    # Calculate the width and height based on the proportions
    width = (orig_right_proportion - orig_left_proportion) * image_width
    height = (orig_bottom_proportion - orig_top_proportion) * image_height

    left_proportion = orig_left_proportion
    top_proportion = orig_top_proportion
    right_proportion = orig_right_proportion
    bottom_proportion = orig_bottom_proportion

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

        reset_orig_left_proportion = x / image.width
        reset_orig_top_proportion = y / image.height
        reset_orig_right_proportion = (x + w) / image.width
        reset_orig_bottom_proportion = (y + h) / image.height

    if a == 0:
        # original crop values, starting here zooming later
        left_proportion = orig_left_proportion
        top_proportion = orig_top_proportion
        right_proportion = orig_right_proportion
        bottom_proportion = orig_bottom_proportion
        print("Original corp")
    
    # Check if zooming in will result in width or height less than 50 pixels
    elif width - 0.1 * image_width > 50 and height - 0.1 * image_height > 50:
        if (orig_left_proportion + 0.05 < orig_right_proportion 
            and orig_top_proportion + 0.05 < orig_bottom_proportion 
            and orig_right_proportion - 0.05 > orig_left_proportion 
            and orig_bottom_proportion - 0.05 > orig_top_proportion): 

            left_proportion = orig_left_proportion + 0.05  # % from the left
            orig_left_proportion = left_proportion  

            top_proportion = orig_top_proportion + 0.05  # % from the top
            orig_top_proportion = top_proportion  

            right_proportion = orig_right_proportion - 0.05  # % from the left
            orig_right_proportion = right_proportion 

            bottom_proportion = orig_bottom_proportion - 0.05 # % from the top
            orig_bottom_proportion = bottom_proportion
            print("Zooming in")
        else: 
            STOP = True
    else: 
        STOP = True

        
    #elif (orig_left_proportion - 0.05 > 0.0 
    #      and orig_top_proportion - 0.05 > 0.0 
    #      and orig_right_proportion + 0.05 < 1.0 
    #      and orig_bottom_proportion + 0.05 < 1.0): 
    #    # stop if some crop line hits a wall 
    #
    #    left_proportion = orig_left_proportion-0.05  # %from the left
    #    orig_left_proportion = left_proportion  
    #    
    #    top_proportion = orig_top_proportion-0.05  # %from the top
    #    orig_top_proportion = top_proportion  
    #    
    #    right_proportion = orig_right_proportion+0.05  # %from the left
    #    orig_right_proportion = right_proportion 
    #    
    #    bottom_proportion = orig_bottom_proportion+0.05 # %from the top
    #    orig_bottom_proportion = bottom_proportion
    #    print("Zooming out")

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
    if (show_images):
        plt.imshow(cv2.cvtColor(image_orig_crop_rect, cv2.COLOR_BGR2RGB))
        plt.title('Original Image with Crop Rectangle')
        plt.show()

    image = image.crop((left, top, right, bottom))

    # Convert the cropped PIL image back to OpenCV image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_center = image
    # Split the image into two
    image_top = image[:image.shape[0]//2, :]
    image_bottom = image[image.shape[0]//2:image.shape[0], :]

    # Save and display the split images
    #cv2.imwrite(os.path.join(output_dir, f'image_top{filetype}'), image_top)
    #if (show_images):
        #plt.imshow(cv2.cvtColor(image_top, cv2.COLOR_BGR2RGB))
        #plt.show()

    #cv2.imwrite(os.path.join(output_dir, f'image_bottom{filetype}'), image_bottom)
    #if (show_images):
        #plt.imshow(cv2.cvtColor(image_bottom, cv2.COLOR_BGR2RGB))
        #plt.show()

    return image_top, image_bottom, image_center
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

STOP = False

show_images = True

# original crop values, starting here zooming later
reset_orig_left_proportion = 0.20  # %from the left
reset_orig_top_proportion = 0.20  # %from the top
reset_orig_right_proportion = 0.80  # %from the left
reset_orig_bottom_proportion = 0.80  # %from the top

# 1
input_dir = "./KasperiP/KP20230628_3/"  # This is your main folder

# 2
output_dir = "outputimages"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# 3
filetype = ".BMP"

# Initialize lists to store data for the table
file_names = []
top_differences = []
bottom_differences = []
center_differences = []
print(os.listdir(input_dir))

# Get a list of all BMP files in the current directory
bmp_files = [f for f in os.listdir(input_dir) if f.endswith(f'{filetype}')]

print("Starting to process images...")
print("Number of images to precess:", len(bmp_files))
print(bmp_files)

# Process each BMP file
for bmp_file in bmp_files:

    # Reset the values at the start of each loop
    orig_left_proportion = reset_orig_left_proportion  # %from the left
    orig_top_proportion = reset_orig_top_proportion  # %from the top
    orig_right_proportion = reset_orig_right_proportion  # %from the left
    orig_bottom_proportion = reset_orig_bottom_proportion  # %from the top

    # a for original start, False for no zooming yet # remember to add image_top, image_bottom if wanted and added above
    image_top, image_bottom, image_center = cutted_picture(a) 

    # Process each image and store the differences
    top_difference = process_image(image_top, f'{bmp_file} Top')
    bottom_difference = process_image(image_bottom, f'{bmp_file} Bottom')
    center_difference = process_image(image_center, f'{bmp_file} Center')
    
    # Add data to the lists
    file_names.append(bmp_file)
    center_differences.append(center_difference)
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
center_differences = [
    kerroin * x if x is not None else 0 for x in center_differences]

# Create a dictionary to associate file names with data dictionaries
file_data = {}

# Populate the file_data dictionary
for file_name, d1, d2, d3 in zip(file_names, top_differences, bottom_differences, center_differences):
    file_data[file_name] = {"top_differences": d1, "bottom_differences": d2, "Center difference": d3}

# Sort the file names
sorted_file_names = sorted(file_names)

# Access the sorted file names and their corresponding data
for file_name in sorted_file_names:
    data_dict = file_data[file_name]

# Create a list to store the table rows
table = []

# Populate the table rows
for file_name, d1, d2, d3 in zip(file_names, top_differences, bottom_differences, center_differences):
    table.append([file_name, d1, d2, d3])

# Sort the table rows based on the file names
sorted_table = sorted(table, key=lambda x: x[0])

# Generate the table
headers = ["File Name", "Top Difference", "Bottom Difference", "Center difference"]
table_str = tabulate(sorted_table, headers, tablefmt="grid")

# Print the table
print(table_str)

# Assuming you have the lists top_differences, bottom_differences, center_differences, and file_names defined as in your code


# Plotting the data
plt.figure(figsize=(15, 6))  # Adjusted width to 15 units
plt.plot(file_names, top_differences, label='Top Differences', marker='o')
plt.plot(file_names, bottom_differences, label='Bottom Differences', marker='o')
plt.plot(file_names, center_differences, label='Center Differences', marker='o')

# Adding labels and title
plt.xlabel('File Name/Time/Item')
plt.ylabel('Difference Value')
plt.title('Differences vs File Name')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()  # Display the legend

# Display the plot
plt.tight_layout()
plt.grid(True)
plt.show()

def moving_average(data, window_size=3):
    """Compute moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Define a window size for the moving average
window = 5

# Compute moving averages
smoothed_top = moving_average(top_differences, window)
smoothed_bottom = moving_average(bottom_differences, window)
smoothed_center = moving_average(center_differences, window)

# Adjust file names for the reduced size after moving average
smoothed_file_names = file_names[window-1:]

# Plotting the data
plt.figure(figsize=(15, 6))
plt.plot(smoothed_file_names, smoothed_top, label='Smoothed Top Differences', marker='o')
plt.plot(smoothed_file_names, smoothed_bottom, label='Smoothed Bottom Differences', marker='o')
plt.plot(smoothed_file_names, smoothed_center, label='Smoothed Center Differences', marker='o')

# Adding labels and title
plt.xlabel('File Name/Time/Item')
plt.ylabel('Difference Value')
plt.title('Smoothed Differences vs File Name')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()  # Display the legend

# Display the plot
plt.tight_layout()
plt.grid(True)
plt.show()

#END table generation