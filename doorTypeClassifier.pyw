import os
from tkinter import *
from PIL import Image, ImageTk
from shutil import move

# Set up the root window
root = Tk()
root.title("Image Classifier")

# Create labels and buttons
image_label = Label(root)

# Define image folder paths
currentDirectory = os.path.dirname(os.path.realpath(__file__))
input_folder = fr'C:\Users\alaci\OneDrive - Arrow Glass Industries\Documents\Scripts\Test\imgRec\trainingData\doorType\test\door\SP'

# Create a list of categories
categories = ["A", "B", "C", "CTM"]  # Add more categories as needed

# Create subfolders if they don't exist
for category in categories:
    category_folder = os.path.join(input_folder, category)
    os.makedirs(category_folder, exist_ok=True)

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

def on_category_click(category_index):
    move_image(categories[category_index])

def move_image(category):
    global current_image_index
    source_path = os.path.join(input_folder, image_files[current_image_index])
    target_path = os.path.join(input_folder, category, image_files[current_image_index])
    move(source_path, target_path)
    
    # Load the next image or close the app if all images are processed
    current_image_index += 1
    if current_image_index < num_images:
        load_image(current_image_index)
    else:
        root.destroy()

def on_key_press(event):
    global current_image_index  # Make current_image_index global
    if event.keysym.isdigit() and 1 <= int(event.keysym) <= len(categories):
        on_category_click(int(event.keysym) - 1)
    elif event.keysym == 'z' or event.keysym == 'Z':
        move_back_to_input_folder()

def move_back_to_input_folder():
    global current_image_index  # Make current_image_index global
    if current_image_index > 0:
        for category in categories:
            source_path = os.path.join(input_folder, category, image_files[current_image_index - 1])
            if os.path.exists(source_path):
                target_path = os.path.join(input_folder, image_files[current_image_index - 1])
                move(source_path, target_path)
                current_image_index -= 1
                load_image(current_image_index)
                break

# Bind the key event handler to the root window
root.bind('<Key>', on_key_press)

# Configure buttons' click actions
buttons = []
for i, category in enumerate(categories):
    button = Button(root, text=category.upper(), command=lambda idx=i: on_category_click(idx))
    buttons.append(button)
    button.grid(row=1, column=i)

# Pack UI elements
image_label.grid(row=0, columnspan=len(categories), padx=20, pady=20)  # Span columns
for i, button in enumerate(buttons):
    button.grid(row=2, column=i, padx=10, pady=10)

# Load and display the first image
load_image(current_image_index)

# Start the GUI event loop
root.mainloop()
