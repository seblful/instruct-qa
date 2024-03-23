import os
from PIL import Image
import numpy as np
import random

import fitz
from pypdf import PdfReader

from tqdm import tqdm


def get_avail_instr_images(save_listdir):
    avail_instr_images = {}

    for image_name in save_listdir:
        image_name = os.path.splitext(image_name)[0]
        instr_name = image_name[:-2]
        image_index = int(image_name[-1])

        avail_instr_images[instr_name] = avail_instr_images.get(
            instr_name, []) + [image_index]

    return avail_instr_images


def get_instr_len_images(instr_dir, instr_listdir):
    instr_len_images = {}

    for pdf_path in instr_listdir:
        full_pdf_path = os.path.join(instr_dir, pdf_path)
        pdf_file = fitz.open(full_pdf_path)
        instr_len_images[pdf_path] = len(pdf_file)

    return instr_len_images


def create_train_data(instr_dir,
                      save_dir,
                      num_images):
    instr_listdir = [instr for instr in os.listdir(
        instr_dir) if instr.endswith('pdf')]

    save_listdir = [instr for instr in os.listdir(
        save_dir)]

    avail_instr_images = get_avail_instr_images(save_listdir=save_listdir)

    instr_len_images = get_instr_len_images(instr_dir, instr_listdir)

    print(instr_len_images)

    # num_saved_images = 0

    # while num_images != num_saved_images:

    #     # Take random instruction
    #     rand_instr = random.choice(instr_listdir)
    #     rand_instr_name = os.path.splitext(rand_instr)[0]
    #     rand_instr_path = os.path.join(instr_dir, rand_instr)
    #     rand_instr_pdf = PdfReader(rand_instr_path)

    #     # Take random instruction page and image
    #     rand_page = random.choice(rand_instr_pdf.pages)
    #     rand_page_ind = rand_instr_pdf.pages.index(rand_page)
    #     rand_image = rand_page.images[0].image

    #     rand_image_name = f"{rand_instr_name}_{rand_page_ind}.png"
    #     rand_image_path = os.path.join(save_dir, rand_image_name)

    #     if not os.path.exists(rand_image_path):
    #         # rand_image.show()
    #         # if rand_image.mode != "RGB":
    #         #     rand_image = rand_image.convert('RGB')

    #         # if rand_image.mode == 'CMYK':
    #         #     imgData = np.array(rand_image)
    #         #     # Inverting the CMYK data
    #         #     invData = 255 - imgData
    #         #     img = Image.fromarray(invData, mode='CMYK')
    #         #     rand_image = rand_image.convert('RGB')

    #         rand_image.save(rand_image_path, "PNG")
    #         num_saved_images += 1
    #         print(f"It was saved {num_saved_images}/{num_images} images.")

# import fitz  # PyMuPDF
# import io
# from PIL import Image

# # Open the PDF file
# pdf_file = fitz.open('your_file.pdf')

# # Iterate over PDF pages
# for page_num in range(len(pdf_file)):
#     # Get the page
#     page = pdf_file[page_num]

#     # Iterate over page images
#     for img_index, img in enumerate(page.get_images(full=True)):
#         # Extract image bytes
#         base_image = pdf_file.extract_image(img[0])
#         image_bytes = base_image["image"]

#         # Get the image extension
#         image_ext = base_image["ext"]

#         # Load it to PIL Image and save it
#         image = Image.open(io.BytesIO(image_bytes))
#         image.save(f"image{page_num + 1}_{img_index + 1}.{image_ext}")

# # Close the PDF after extraction
# pdf_file.close()
