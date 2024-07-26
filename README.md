# Process_Optimizers

Scripts written by Ryan Aday to optimize LBPF prints.

## Before you run:

Type the command:

    pip install -r requirements.txt

## parameter_opt.py
Optimizes laser power, scanning speed, hatch spacing, and layer thickness for the LBPF process.

Assumes target normalized enthalpy of 30, constant beam diameter.
Requires material properties.
Constrained by melt pool depth, hatch distance (see papers).
Uses the Sequential Least Squares algorithm (SLS) to handle constraints.

Example output:

    Optimal Laser Power: 200.00 W
    Optimal Scan Speed: 1000.00 mm/s
    Optimal Hatch Spacing: 0.10 mm
    Optimal Layer Thickness: 0.03 mm

## powder_hough.py
Observes a zoomed-in image of powder. Uses OpenCV to identify and characterize each powder molecule.
Output is a mean and standard deviation of the molecule radii in units of pixels.
Currently tuned to powder.png, but it should also work fairly well for similar images.

Binarization and canny edge detection for contours:

![Figure_1](https://github.com/user-attachments/assets/563e14b1-7dc9-47f3-8143-c7159224597d)

Output image:

![output_image](https://github.com/user-attachments/assets/4eda6586-5d06-41aa-9e68-2c5061e1c13e)

## process_control.py
Takes in any JSON or CSV file and provides plots to show you if a process is in or out of control.
Asks for user input for file path.
Outputs plot with UCL and LCL. In console, if process out of control samples are indicated and a new target is suggested to compensate.

Example Plot:

![Figure_2](https://github.com/user-attachments/assets/e929b244-9500-433f-b560-3aa6c61bc144)

Example Output:

    Enter the path to the CSV or JSON file: out_of_control.csv
    The process for Temperature is out of control at points: [96, 97, 98, 99]
    Suggested new target for Temperature: 20.25751161336211
    The process for Humidity is out of control at points: [95]
    Suggested new target for Humidity: 49.34864012128788

## Good Links:
 - https://answers.opencv.org/question/212287/fit-ellipse-with-most-points-on-contour-instead-of-least-squares/
 - https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
 - https://www.geeksforgeeks.org/python-opencv-canny-function/
 - https://cvexplained.wordpress.com/2020/05/18/morphological-operations/
 - https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
 - https://www.educba.com/opencv-approxpolydp/
 - https://medium.com/@isinsuarici/hough-circle-transform-parameter-tuning-with-examples-6b63478377c9
 - https://www.sciencedirect.com/science/article/pii/S221486041830188X
 - https://www.sciencedirect.com/science/article/abs/pii/S0272884222011713
 - https://www.nature.com/articles/s41598-024-63288-1
