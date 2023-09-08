import os
from tkinter import *
from PIL import Image, ImageTk
from shutil import move

# Set up the root window
root = Tk()
root.title("Image Classifier")

# Create labels and buttons
image_label = Label(root)
door_button = Button(root, text="Door")
panel_button = Button(root, text="Panel")

# Define image folder paths
input_folder = r'C:\Users\alaci\OneDrive - Arrow Glass Industries\Documents\Scripts\Test\imgRec\trainingData\doorType\test'
door_folder = os.path.join(input_folder, "door")
panel_folder = os.path.join(input_folder, "panel")

# Create subfolders if they don't exist
os.makedirs(door_folder, exist_ok=True)
os.makedirs(panel_folder, exist_ok=True)

# Get a list of image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith((".jpg", ".png", ".jpeg"))]
num_images = len(image_files)
current_image_index = 0

def load_image(index):
    global current_image_index  # Make current_image_index global
    image_path = os.path.join(input_folder, image_files[index])
    img = Image.open(image_path)
    img.thumbnail((600, 600))  # Larger image dimensions
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

def on_door_click():
    move_image("door")

def on_panel_click():
    move_image("panel")

def move_image(category):
    global current_image_index
    source_path = os.path.join(input_folder, image_files[current_image_index])
    target_path = os.path.join(door_folder if category == "door" else panel_folder, image_files[current_image_index])
    move(source_path, target_path)
    
    # Load the next image or close the app if all images are processed
    current_image_index += 1
    if current_image_index < num_images:
        load_image(current_image_index)
    else:
        root.destroy()

def on_key_press(event):
    global current_image_index  # Make current_image_index global
    if event.keysym == '1':
        on_door_click()
    elif event.keysym == '2':
        on_panel_click()
    elif event.keysym == 'z' or event.keysym == 'Z':
        move_back_to_input_folder()

def move_back_to_input_folder():
    global current_image_index  # Make current_image_index global
    if current_image_index > 0:
        source_path = os.path.join(door_folder if "door" in image_files[current_image_index - 1] else panel_folder, image_files[current_image_index - 1])
        target_path = os.path.join(input_folder, image_files[current_image_index - 1])
        
        # Check if the source_path exists before moving
        if os.path.exists(source_path):
            move(source_path, target_path)
            
            # Load the previous image
            current_image_index -= 1
            load_image(current_image_index)
        else:
            print(f"Source image '{source_path}' not found.")


# Bind the key event handler to the root window
root.bind('<Key>', on_key_press)

# Configure buttons' click actions
door_button.config(command=on_door_click)
panel_button.config(command=on_panel_click)

# Pack UI elements
image_label.grid(row=0, columnspan=2, padx=20, pady=20)  # Span two columns
door_button.grid(row=1, column=0)  # First column, no padx or pady
panel_button.grid(row=1, column=1)  # Second column, no padx or pady

# Load and display the first image
load_image(current_image_index)

# Start the GUI event loop
root.mainloop()
