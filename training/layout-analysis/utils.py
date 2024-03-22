import os
import random

from pypdf import PdfReader

from tqdm import tqdm


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
        rand_instr_pdf = PdfReader(rand_instr_path)

        # Take random instruction page and image
        rand_page = random.choice(rand_instr_pdf.pages)
        rand_page_ind = rand_instr_pdf.pages.index(rand_page)
        rand_image = rand_page.images[0].image
        rand_image_name = f"{rand_instr_name}_{rand_page_ind}.jpg"
        rand_image_path = os.path.join(save_dir, rand_image_name)

        if not os.path.exists(rand_image_path):
            rand_image.save(rand_image_path)
            num_saved_images += 1
            print(f"It was saved {num_saved_images}/{num_images} images.")
