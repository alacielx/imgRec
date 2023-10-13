import cv2
import os

# Function to handle mouse events for cropping
def crop_image(event, x, y, flags, param):
    
    global ref_point, cropping, image, roi_name

    image_display = cv2.resize(image, (600, 800))
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        
        # Make sure smallest number is first to crop image correctly
        x1 = min(ref_point[0][0], ref_point[1][0])
        y1 = min(ref_point[0][1], ref_point[1][1])
        x2 = max(ref_point[0][0], ref_point[1][0])
        y2 = max(ref_point[0][1], ref_point[1][1])
        
        ref_point = [[x1, y1],
                     [x2, y2]]

        # Draw a rectangle around the region of interest on the displayed image
        cv2.rectangle(image_display, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow(roi_name, image_display)

        # Calculate resizing factor
        resize_factor_x = original_image.shape[1] / image_display.shape[1]
        resize_factor_y = original_image.shape[0] / image_display.shape[0]

        # Adjust ROI coordinates based on resizing
        x1 = int(ref_point[0][0] * resize_factor_x)
        y1 = int(ref_point[0][1] * resize_factor_y)
        x2 = int(ref_point[1][0] * resize_factor_x)
        y2 = int(ref_point[1][1] * resize_factor_y)

        # Crop and save the region of interest on the original image
        if len(ref_point) == 2:
            roi = original_image[y1:y2, x1:x2]
            cv2.imwrite(cropped_path, roi)
            print(f'Saved: {cropped_filename}')
            # cv2.imshow("Cropped", roi)
            # cv2.waitKey(0)


# Define the directory containing your images
image_directory = r'C:\Users\agarza\OneDrive - Arrow Glass Industries\Documents\Scripts\Test\imgRec\trainingData\doorType\train - Copy\CTM'

# Define the directory where cropped sections will be saved
output_directory = r'C:\Users\agarza\OneDrive - Arrow Glass Industries\Documents\Scripts\Test\imgRec\trainingData\doorType\train - Copy\output'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Define the names for the ROIs
roi_names = ["Height1", "Height2", "Width1", "Width2", "HandleSide", "HingeSide", "Bottom"]

# Iterate through the images in the image directory
for roi_name in roi_names:
    for image_filename in os.listdir(image_directory):
        cropped_filename = f'{roi_name}_{image_filename}'
        cropped_path = os.path.join(output_directory, cropped_filename)
        if (image_filename.endswith('.jpg') or image_filename.endswith('.png')) and not os.path.exists(cropped_path):
            # Load the original image
            image_path = os.path.join(image_directory, image_filename)
            original_image = cv2.imread(image_path)
            image = original_image.copy()  # Make a copy to work with
            
            # Create a window to display the image
            cv2.namedWindow(roi_name)
            cv2.setMouseCallback(roi_name, crop_image)

            ref_point = []
            cropping = False
            roi_idx = 0

            print(f"Select ROI for '{roi_name}'")
            image_display = cv2.resize(image, (600, 800))
            cv2.imshow(roi_name, image_display)
                
            while True:
                # Resize the image for better display (adjust the dimensions as needed)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(" "):
                    break

            cv2.destroyAllWindows()