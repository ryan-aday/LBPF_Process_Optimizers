import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Read the image
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image is None:
        print("Error: Unable to read image")
        return None
        
    ret, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)


    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(image, (25, 25), 1.5)

    # Apply Canny edge detector to find sharp differences
    edges = cv2.Canny(blurred, 100, 250)

    # Perform morphological operations to remove small noise and close gaps
    #edges = thresh
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    #morph = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    #morph = cv2.erode(morph, kernel, iterations=1)
    #morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
       
    morph = edges
       
    #morph = cv2.dilate(morph, kernel, iterations=1)
 
    #morph = thresh
 
    return morph

def find_circles(image_path):
    # Preprocess the image
    thresh = preprocess_image(image_path)
    if thresh is None:
        return []

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Filter out nested contours
    filtered_contours = [cnt for i, cnt in enumerate(contours) if hierarchy[0][i][3] == -1]
    #filtered_contours = contours
    circles = []
    for cnt in filtered_contours:
        cnt = cv2.approxPolyDP(cnt, 0.01* cv2.arcLength(cnt, True), True)
        if len(cnt) >= 10:
            # Compute circularity
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity > 0.0 and area >= 10:  # Circularity threshold
            #if area >= 10:
                ellipse = cv2.fitEllipse(cnt)
                circles.append(ellipse)

    return circles, filtered_contours, thresh

def draw_circles(image_path, circles, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image")
        return

    height, width = image.shape[:2]
    max_axis_size = min(width, height) / 3

    for circle in circles:
        center, axes, angle = circle
        major_axis, minor_axis = max(axes), min(axes)
        # Filter out very small circles that are likely noise and those greater than a third of the image size
        if 20 < major_axis < max_axis_size and 20 < minor_axis < max_axis_size:
            cv2.ellipse(image, circle, (0, 0, 255), 2)  # Draw circle in red

    # Save the output image
    cv2.imwrite(output_path, image)

def calculate_statistics(circles, image_shape):
    height, width = image_shape[:2]
    max_axis_size = min(width, height) / 3

    major_axes = [max(circle[1]) for circle in circles if 20 < max(circle[1]) < max_axis_size and 10 < min(circle[1]) < max_axis_size]
    minor_axes = [min(circle[1]) for circle in circles if 20 < max(circle[1]) < max_axis_size and 10 < min(circle[1]) < max_axis_size]

    mean_major = np.mean(major_axes)
    std_major = np.std(major_axes)
    mean_minor = np.mean(minor_axes)
    std_minor = np.std(minor_axes)

    return mean_major, std_major, mean_minor, std_minor

def main(image_path, output_path):
    circles, filtered_contours, thresh_image = find_circles(image_path)
    if not circles:
        print("No circles detected.")
        return

    mean_major, std_major, mean_minor, std_minor = calculate_statistics(circles, thresh_image.shape)

    print(f"Mean Major Axis: {mean_major:.2f}, Standard Deviation: {std_major:.2f}")
    print(f"Mean Minor Axis: {mean_minor:.2f}, Standard Deviation: {std_minor:.2f}")

    draw_circles(image_path, circles, output_path)

    # Display the binarized image with contours
    plt.imshow(thresh_image, cmap='gray')
    for cnt in filtered_contours:
        plt.plot(cnt[:, 0, 0], cnt[:, 0, 1], color='orange')
    plt.title(f"Mean Major: {mean_major:.2f}, Std: {std_major:.2f}\nMean Minor: {mean_minor:.2f}, Std: {std_minor:.2f}")
    plt.show()

image_path = 'powder.png'
output_path = 'output_image.jpg'
main(image_path, output_path)
