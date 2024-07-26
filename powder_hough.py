import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Unable to read image")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to smooth the image
    image = cv2.GaussianBlur(image, (7, 7), 1.2)
    
    # Apply Canny edge detector to find sharp differences
    image = cv2.Canny(image, 50, 180)
    
    return image

def find_circles(image_path):
    gray = preprocess_image(image_path)
    if gray is None:
        return [], None

    h, w = gray.shape[:]

    # Detect circles in the image
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1.52, 
        minDist=int(w / 20), 
        param1=10, 
        param2=40, 
        minRadius=int(w / 80), 
        maxRadius=int(w / 17.5),
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    return circles, gray

def draw_circles(image_path, circles, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image")
        return

    if circles is not None:
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 0, 255), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # Save the output image
    cv2.imwrite(output_path, image)

def calculate_statistics(circles):
    if circles is None:
        return np.nan, np.nan

    radii = [r for (_, _, r) in circles]
    mean_radius = np.mean(radii)
    std_radius = np.std(radii)

    return mean_radius, std_radius

def main(image_path, output_path):
    circles, gray_image = find_circles(image_path)
    if circles is None:
        print("No circles detected.")
        return

    mean_radius, std_radius = calculate_statistics(circles)

    print(f"Mean Radius: {mean_radius:.2f}, Standard Deviation: {std_radius:.2f}")

    draw_circles(image_path, circles, output_path)

    # Display the binarized image with circles
    plt.imshow(gray_image, cmap='gray')
    if circles is not None:
        for (x, y, r) in circles:
            plt.gca().add_patch(plt.Circle((x, y), r, color='r', fill=False))
    plt.title(f"Mean Radius: {mean_radius:.2f}, Std: {std_radius:.2f}")
    plt.show()

image_path = 'powder.png'
output_path = 'output_image.jpg'
main(image_path, output_path)
