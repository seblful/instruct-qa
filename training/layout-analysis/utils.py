import os
import io
from PIL import Image
import numpy as np
import random

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


def create_train_data(instr_dir,
                      save_dir,
                      num_images):
    instr_listdir = [instr for instr in os.listdir(
        instr_dir) if instr.endswith('pdf')]

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
