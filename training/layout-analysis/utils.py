import os

import cv2
import numpy as np
import random

from PIL import Image
import io
import fitz


def get_image(pdf_file, pdf_page):
    # Get image from pdf page and transform it to bytes image
    fitz_image = pdf_page.get_images(full=False)[0]
    base_image = pdf_file.extract_image(fitz_image[0])
    bytes_image = base_image["image"]

    # Get the image extension
    image_ext = base_image["ext"]

    # Load it to PIL Image and reformat it
    image = Image.open(io.BytesIO(bytes_image))

    if image_ext != 'jpg':
        image = image.convert('RGB')

    return image


def resize_image(image, new_width=None, new_height=None):
    # Get the original image dimensions
    height, width = image.shape[:2]

    if new_width is None and new_height is None:
        return image  # No resizing needed

    if new_width is None:
        # Calculate the aspect ratio and resize based on the new height
        aspect_ratio = new_height / height
        new_width = int(width * aspect_ratio)
    else:
        # Calculate the aspect ratio and resize based on the new width
        aspect_ratio = new_width / width
        new_height = int(height * aspect_ratio)

    # Perform the resizing while maintaining the aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


def create_train_data(instr_dir,
                      save_dir,
                      num_images):
    # Listdir with instructions
    instr_listdir = [instr for instr in os.listdir(
        instr_dir) if instr.endswith('pdf')]

    # Counter for saved images
    num_saved_images = 0

    while num_images != num_saved_images:

        # Take random instruction
        rand_instr = random.choice(instr_listdir)
        rand_instr_name = os.path.splitext(rand_instr)[0]
        rand_instr_path = os.path.join(instr_dir, rand_instr)
        rand_instr_pdf = fitz.open(rand_instr_path)

        # Take random instruction page and image
        rand_page_ind = random.randint(0, len(rand_instr_pdf) - 1)
        rand_page = rand_instr_pdf[rand_page_ind]

        rand_image = get_image(pdf_file=rand_instr_pdf,
                               pdf_page=rand_page)

        rand_image_name = f"{rand_instr_name}_{rand_page_ind}.jpg"
        rand_image_path = os.path.join(save_dir, rand_image_name)

        if not os.path.exists(rand_image_path):
            rand_image.save(rand_image_path, "JPEG")
            num_saved_images += 1
            print(f"It was saved {num_saved_images}/{num_images} images.")

        rand_instr_pdf.close()


def create_train_data_manually(instr_dir,
                               save_dir):
    # Listdir with instructions
    instr_listdir = [instr for instr in os.listdir(
        instr_dir) if instr.endswith('pdf')]

    while True:
        # Take random instruction
        random.seed()
        rand_instr = random.choice(instr_listdir)
        rand_instr_name = os.path.splitext(rand_instr)[0]
        rand_instr_path = os.path.join(instr_dir, rand_instr)
        rand_instr_pdf = fitz.open(rand_instr_path)

        # Iterating over pages in instruction
        for page_num in range(rand_instr_pdf.page_count):
            page = rand_instr_pdf[page_num]
            image = get_image(pdf_file=rand_instr_pdf,
                              pdf_page=page)

            # Show image
            image_to_show = resize_image(np.array(image), new_height=1000)
            cv2.imshow("PDF Image", image_to_show)

            # Handle pressed keys
            pressed_key = cv2.waitKey(0) & 0xFF

            # Save image if S is pressed
            if pressed_key in [83, 115, 219, 251]:
                # Create name of image
                image_name = f"{rand_instr_name}_{page_num}.jpg"
                image_path = os.path.join(save_dir, image_name)
                if not os.path.exists(image_path):
                    image.save(image_path, "JPEG")

            # Skip one instruction if B is pressed
            elif pressed_key in [66, 98, 200, 232]:
                break

            # Exit if Q is pressed
            elif pressed_key in [113, 81, 201, 233]:
                return

            else:
                continue
