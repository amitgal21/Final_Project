from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import Label, Button, filedialog

w = Tk()
w.geometry('1400x700')
w.configure(bg='#262626')
w.resizable(0, 0)
w.title('Toggle Menu')

# List to store the labels that are created
labels_list = []


def clear_labels():
    # Destroys all labels in the list and clears the list
    for label in labels_list:
        label.destroy()
    labels_list.clear()


def show_translated_text():
    clear_labels()  # Clears the existing labels

    # Adding the title label
    title_label = Label(w,
                        text="Bacteria Prediction & Estimation",
                        fg='white', bg='#262626', font=('Comic Sans MS', 20, 'bold'))
    title_label.place(relx=0.63, rely=0.2, anchor='center')  # Adjust placement as needed

    # Adding the detailed description label below the title
    translated_label = Label(w,
                             text="Welcome to Bactivision.\n Our system predicts bacteria from Gram stain images, "
                                  "using AI and image processing.\n We utilize models like AlexNet and VGG16, "
                                  "trained to identify bacteria types.\n The system assesses dataset quality, "
                                  "classifies bacteria into Gram-positive or Gram-negative,\n "
                                  "and accurately identifies species with 70-80% precision.\n "
                                  "Our Unet-based architecture excels at analyzing microscopic objects, "
                                  "like bacteria and parasites.\n Advanced algorithms, like Otsu's, "
                                  "estimate bacteria quantity, enhancing prediction accuracy.",
                             fg='white', bg='#262626', font=('Comic Sans MS', 16), wraplength=w.winfo_width() - 350)
    translated_label.place(relx=0.6, rely=0.5, anchor='center')  # Adjust placement as needed


    labels_list.append(title_label)
    labels_list.append(translated_label)


def predict_bacteria_details():
    clear_labels()  # Clear existing labels

    selected_files = []  # List to store the paths of selected images
    file_labels = []  # List to keep track of file name labels for UI update
    image_labels = []  # List to keep track of image labels for UI update
    detail_labels = []  # List to keep track of detail labels for UI update

    # Clears file name, image, and detail labels before updating the UI
    def clear_file_and_image_labels():
        for label in file_labels + image_labels + detail_labels:
            label.destroy()
        file_labels.clear()
        image_labels.clear()
        detail_labels.clear()

    # Instruction label for uploading photos
    instruction_label = Label(w,
                              text="Please upload up to 3 Gram stain images of bacteria.\n"
                                   "Our system will predict the bacteria family based on the images.",
                              fg='white', bg='#262626', font=('Comic Sans MS', 16), wraplength=w.winfo_width() - 350)
    instruction_label.place(relx=0.6, rely=0.1, anchor='center')
    labels_list.append(instruction_label)

    # Function to update the user interface (UI) with the selected images and their details.
    def update_ui():
        # Update the text of the files_label to show the number of images currently selected out of a maximum of 3.
        files_label['text'] = f"Images selected: {len(selected_files)}/3"

        # Call the function to clear any existing labels related to file names, images, and detail information.
        clear_file_and_image_labels()

        # Calculate the horizontal spacing and starting position for image labels to center them dynamically based on the number of selected images.
        spacing = 0.2  # Determines the spacing between image labels.
        start_pos = 0.6 - ((
                                   len(selected_files) - 1) * spacing / 2)  # Adjusts the starting position to center align the images.

        # Iterate through the list of selected files to process and display each one.
        for i, filename in enumerate(selected_files):
            # Extract the base file name and the initial part of the file name (assumed to be the bacteria name).
            file_name = filename.split('/')[-1]  # Extracts just the file name without the path.
            bacteria_name = file_name.split('_')[
                0]

            # Predict the Gram stain color (e.g., positive or negative) of the bacteria. This requires an external function.
            predicted_class = predict_gram_color(filename)  # Assumes this function exists and returns a string.

            # Determine the shape of the bacteria based on its name, using another assumed function.
            bacteria_shape = find_bacteria_shape(bacteria_name)

            # Calculate the coverage of bacteria in the image, which is a metric provided by another assumed function.
            bacteria_coverage = find_the_size_of_bacteria(filename)

            # Create and place a label with the bacteria name at the calculated position.
            file_label = Label(w, text=f"{bacteria_name}", fg='white', bg='#262626', font=('Comic Sans MS', 10))
            file_label.place(relx=start_pos + i * spacing, rely=0.30, anchor='center')
            file_labels.append(file_label)

            # Load the image, resize it to a thumbnail, and display it.
            img = Image.open(filename)
            img.thumbnail((100, 100), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            img_label = Label(w, image=photo, bg='#262626')
            img_label.image = photo  # Keep a reference to prevent garbage collection.
            img_label.place(relx=start_pos + i * spacing, rely=0.4, anchor='center')
            image_labels.append(img_label)


            details = [f"Gram Type: {predicted_class}", f"Shape: {bacteria_shape}",
                       f"Bacteria Coverage: {bacteria_coverage}%",
                       f"Bacteria Name: {bacteria_name}"]
            for j, detail in enumerate(details):
                detail_label = Label(w, text=detail, fg='white', bg='#262626', font=('Comic Sans MS', 10))
                detail_label.place(relx=start_pos + i * spacing, rely=0.45 + j * 0.05, anchor='center')
                detail_labels.append(detail_label)

    # Function to handle the file upload action.
    def upload_action():
        # Allows selection of up to 3 images, opening a file dialog to choose an image file.
        if len(selected_files) < 3:
            filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")])
            if filename:  # If a file was selected, add it to the list and update the UI.
                selected_files.append(filename)
                print(f"File selected: {filename}")
                update_ui()
        else:
            print("Maximum of 3 images are already selected.")

    # Setup for the upload button and the label displaying the number of selected images.
    upload_button = Button(w, text="Upload Photo", command=upload_action,
                           fg='white', bg='#0f9d9a', font=('Comic Sans MS', 12))
    upload_button.place(relx=0.6, rely=0.2, anchor='center')
    labels_list.append(upload_button)

    files_label = Label(w, text="Images selected: 0/3", fg='white', bg='#262626', font=('Comic Sans MS', 12))
    files_label.place(relx=0.6, rely=0.25, anchor='center')
    labels_list.append(files_label)


def predict_gram_color(image_path):
    from keras.models import load_model
    from keras.preprocessing.image import load_img, img_to_array
    import numpy as np

    model_path = r'C:\Users\amitg\PycharmProjects\toggle_win\trained_vgg16_model.h5'

    model = load_model(model_path)

    def preprocess_image(image, target_size=(224, 224)):
        img = load_img(image, target_size=target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img

    new_image = preprocess_image(image_path)
    prediction = model.predict(new_image)
    predicted_class = 'Purple' if prediction[0][0] < 0.5 else 'Red'
    return predicted_class


def find_bacteria_shape(bacteria_names, file_path='names.txt'):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Splitting each line by '-' to separate the bacteria name and its shape
                parts = line.strip().split('-')
                if len(parts) == 2:
                    name, shapes = parts
                    if name.strip() == bacteria_names:
                        return shapes.strip()
            # If the bacteria name was not found in the file
            return f"The shape of '{bacteria_names}' was not found in the file."
    except FileNotFoundError:
        return "The specified file was not found."


def find_the_size_of_bacteria(image_path):
    import numpy as np
    image = Image.open(image_path)
    image_gray = image.convert('L')  # Convert to grayscale
    image_np = np.array(image_gray)

    # Simple thresholding to separate bacteria from background
    threshold = image_np.mean()
    bacteria_pixels = image_np < threshold

    # Calculate coverage percentage and round it to 3 decimal places
    bacteria_coverage = np.sum(bacteria_pixels) / bacteria_pixels.size * 100
    bacteria_coverage_rounded = round(bacteria_coverage, 3)

    return bacteria_coverage_rounded


def estimate_dataset():
    clear_labels()  # Clear existing labels before displaying a new image

    # 1
    image_path_first = 'C:/Users/amitg/PycharmProjects/toggle_win/datasetH.jpg'
    img_first = Image.open(image_path_first)
    img_first = img_first.resize((300, 300), Image.Resampling.LANCZOS)
    img_photo_first = ImageTk.PhotoImage(img_first)
    image_label_first = Label(w, image=img_photo_first, bg='#262626')
    image_label_first.image = img_photo_first
    image_label_first.place(relx=0.4, rely=0.3, anchor='center')
    labels_list.append(image_label_first)

    # 2
    image_path_second = 'C:/Users/amitg/PycharmProjects/toggle_win/bacteria_area.jpg'
    img_second = Image.open(image_path_second)
    img_second = img_second.resize((300, 300), Image.Resampling.LANCZOS)
    img_photo_second = ImageTk.PhotoImage(img_second)
    image_label_second = Label(w, image=img_photo_second, bg='#262626')
    image_label_second.image = img_photo_second
    image_label_second.place(relx=0.7, rely=0.3, anchor='center')
    labels_list.append(image_label_second)

    # 3
    image_path_third = 'C:/Users/amitg/PycharmProjects/toggle_win/corultion.jpg'
    img_third = Image.open(image_path_third)
    img_third = img_third.resize((500, 300), Image.Resampling.LANCZOS)
    img_photo_third = ImageTk.PhotoImage(img_third)
    image_label_third = Label(w, image=img_photo_third, bg='#262626')
    image_label_third.image = img_photo_third
    image_label_third.place(relx=0.7, rely=0.75, anchor='center')
    labels_list.append(image_label_third)


    dataset_size_mb = "8 GB"
    number_of_images = "840"
    number_of_bacteria_types = "33"
    image_resolution = "2048*1052"


    dataset_info_labels = [
        f"Dataset Size: {dataset_size_mb}",
        f"Number of Images: {number_of_images}",
        f"Number of Bacteria Types: {number_of_bacteria_types}",
        f"Image Resolution: {image_resolution}"
    ]

    for i, info in enumerate(dataset_info_labels):
        info_label = Label(w, text=info, fg='white', bg='#262626', font=('Comic Sans MS', 12))
        info_label.place(relx=0.4, rely=0.6 + i * 0.05, anchor='center')
        labels_list.append(info_label)


def pre_process_show():
    from tkinter import font as tkFont
    """Clear existing labels and display images and preprocessing steps summary."""
    clear_labels()

    # Display the first (original) image
    image_path1 = 'C:/Users/amitg/PycharmProjects/toggle_win/preprocess.jpg'
    img1 = Image.open(image_path1)
    img1 = img1.resize((500, 300), Image.Resampling.LANCZOS)
    img_photo1 = ImageTk.PhotoImage(img1)
    image_label1 = Label(w, image=img_photo1, bg='#262626')
    image_label1.image = img_photo1  # Keep reference
    image_label1.place(relx=0.45, rely=0.25, anchor='center')
    labels_list.append(image_label1)

    # Display the second (segmentation) image
    image_path2 = 'C:/Users/amitg/PycharmProjects/toggle_win/segment.jpg'
    img2 = Image.open(image_path2)
    img2 = img2.resize((500, 300), Image.Resampling.LANCZOS)
    img_photo2 = ImageTk.PhotoImage(img2)
    image_label2 = Label(w, image=img_photo2, bg='#262626')
    image_label2.image = img_photo2  # Keep reference
    image_label2.place(relx=0.45, rely=0.7, anchor='center')
    labels_list.append(image_label2)

    summary_font = tkFont.Font(family="Helvetica", size=12, weight="bold")

    # Create and position the Text widget for preprocessing summary with adjusted font
    preprocessing_summary = Text(w, height=28, width=50, bg='#404040', fg='white', wrap=WORD, borderwidth=2,
                                 relief="solid", padx=10, pady=10, font=summary_font)
    preprocessing_summary.place(relx=0.82, rely=0.5, anchor='center')

    # Preprocessing steps summary text
    summary_text = """
The Image Preprocessing Process:   

Directory Traversal: Iterates over subdirectories, identifying image files.

Preprocessing of Original Images: Images are resized to 224x224 pixels.

Conversion to Array: Images are converted to NumPy arrays.

Normalization: Pixel values normalized to the range 0 to 1.

Dimension Expansion for Grayscale Images: Grayscale images are expanded to three channels.

Loading and Preprocessing of Segmentation Images: Similar steps are applied to segmentation images.

Adjustment for Single-channel Images: Segmentation images are adjusted to single-channel.

Data Compilation: Images and segmentations are stored in separate lists and converted to NumPy arrays.

Error Handling: Includes error handling for loading or processing failures.
    """

    # Insert the summary text and make the widget read-only
    preprocessing_summary.insert("1.0", summary_text)
    preprocessing_summary.config(state=DISABLED)
    labels_list.append(preprocessing_summary)


def log_out():
    exit(True)


menu_open = False

f1 = None


def toggle_win():
    global menu_open, f1
    if menu_open:
        f1.place_forget()
        menu_open = False
    else:
        f1 = Frame(w, width=300, height=900, bg='#12c4c0')
        f1.place(x=0, y=0)
        menu_open = True

        def bttn(x, y, text, bcolor, fcolor, cmd):
            def on_entera(e):
                myButton1['background'] = bcolor
                myButton1['foreground'] = '#262626'

            def on_leavea(e):
                myButton1['background'] = fcolor
                myButton1['foreground'] = '#262626'

            myButton1 = Button(f1, text=text, width=42, height=2, fg='#262626',
                               border=0, bg=fcolor, activeforeground='#262626',
                               activebackground=bcolor, command=cmd)
            myButton1.bind("<Enter>", on_entera)
            myButton1.bind("<Leave>", on_leavea)
            myButton1.place(x=x, y=y)

        bttn(0, 80, 'H O M E', '#0f9d9a', '#12c4c0', show_translated_text)
        bttn(0, 120, 'P R E D I C T - B A C T E R I A  - F A M I L Y', '#0f9d9a', '#12c4c0', predict_bacteria_details)
        bttn(0, 160, 'E S T I M A T E - D A T A - SET', '#0f9d9a', '#12c4c0', estimate_dataset)
        bttn(0, 200, 'I M A G E P R E - P R O C E S S', '#0f9d9a', '#12c4c0', pre_process_show)
        bttn(0, 600, 'E X I T', '#0f9d9a', '#12c4c0', log_out)

        def dele():
            global menu_open
            f1.place_forget()
            menu_open = False
            clear_labels()

        global img2
        img2 = ImageTk.PhotoImage(Image.open("close.png"))
        Button(f1, image=img2, border=0, command=dele, bg='#12c4c0', activebackground='#12c4c0').place(x=5, y=10)


img1 = ImageTk.PhotoImage(Image.open("open.png"))
Button(w, image=img1, command=toggle_win, border=0, bg='#262626', activebackground='#262626').place(x=5, y=10)

w.mainloop()
