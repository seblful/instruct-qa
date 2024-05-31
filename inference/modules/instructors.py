import os
import re
import requests
import io
from PIL import Image

import fitz


class Instruction:
    def __init__(self,
                 instr_dir,
                 clean_instr_dir,
                 extr_instr_dir,
                 pdf_path_or_url):

        # Paths
        self.instr_dir = instr_dir
        self.clean_instr_dir = clean_instr_dir
        self.extr_instr_dir = extr_instr_dir
        self.pdf_path_or_url = pdf_path_or_url

        self.base_url = "https://www.rceth.by/NDfiles/instr/"
        self.__pdf_path = None
        self.__pdf_url = None
        self.__md_path = None

        # Regexes for path and url
        self.path_regexp = re.compile(r"^(?!https?:\/\/|www\.).*$")
        self.url_regexp = re.compile(
            r"(https?:\/\/)?(www\.)?rceth\.by[\/]{1,2}NDfiles\/instr\/[0-9_]*_[isp]\.pdf")

        self.instr_pdf = self.open_pdf()

        # Images from instruction
        self.__instr_imgs = None

        self.was_cleaned = os.path.exists(
            os.path.join(self.clean_instr_dir, self.md_path))

    def input_is_path(self):
        if self.path_regexp.match(self.pdf_path_or_url):
            return True
        return False

    @property
    def pdf_path(self):
        if self.__pdf_path is None:
            if self.input_is_path():
                self.__pdf_path = self.pdf_path_or_url

            else:
                self.__pdf_path = os.path.basename(self.pdf_path_or_url)

        return self.__pdf_path

    @property
    def pdf_url(self):
        if self.__pdf_url is None:
            if not self.input_is_path():
                self.__pdf_url = self.pdf_path_or_url

            else:
                base_path = os.path.basename(self.pdf_path_or_url)
                self.__pdf_url = os.path.join(
                    self.base_url, base_path)

        return self.__pdf_url

    @property
    def md_path(self):
        if self.__md_path is None:
            self.__md_path = os.path.splitext(self.pdf_path)[0] + ".md"

        return self.__md_path

    def open_pdf(self):
        # Create full pdf path
        full_pdf_path = os.path.join(self.instr_dir, self.pdf_path)

        # Download instruction if it was not previously downloaded
        if not os.path.exists(full_pdf_path):
            print("Instruction was not downloaded before, downloading instruction...")
            self.download_pdf()

        # Open instruction
        instr_pdf = fitz.open(full_pdf_path)

        return instr_pdf

    def download_pdf(self):
        # Validate url
        if not self.validate_url(pdf_url=self.pdf_url):
            raise ValueError('Url is not valid.')

        # Create full pdf path
        full_pdf_path = os.path.join(self.instr_dir, self.pdf_path)

        # Download instruction
        res = requests.get(self.pdf_url)

        with open(full_pdf_path, 'wb') as f:
            f.write(res.content)

        return None

    def validate_url(self, pdf_url):
        result = self.url_regexp.match(pdf_url)

        if result:
            return True

        return False

    @property
    def instr_imgs(self):
        if self.__instr_imgs is None:
            self.__instr_imgs = self.__extract_images()

        return self.__instr_imgs

    def __extract_images(self):
        # Create empty list to store images
        images = []
        # Iterate through all the pages of the PDF
        for page_index in range(self.instr_pdf.page_count):
            # Get the current page
            page = self.instr_pdf[page_index]
            # Get page rotation
            page_rotation = page.rotation

            # Get the list of images on the current page
            image_list = page.get_images()

            # Iterate through the list of images
            for img in image_list:
                # Extract the image
                base_image = self.instr_pdf.extract_image(img[0])

                # Get the image data and extension
                image_data = base_image["image"]
                image_extension = base_image["ext"]

                # Create an Image object from the image data
                image = Image.open(io.BytesIO(image_data))

                # Change type of image
                if image_extension != 'jpeg':
                    image = image.convert('RGB')

                # Rotate image
                if page_rotation != 0:
                    image = image.rotate(page_rotation)

                # Append image to images list
                images.append(image)

        return images

    def clean_instr(self,
                    image_processor):
        # Create empty list to store images
        cleaned_images = []

        # Iterate through all the images and clean it
        for image in self.instr_imgs:
            # cleaned_img = image_processor.process(image)
            import numpy as np
            cleaned_img = np.array(image)
            cleaned_images.append(cleaned_img)

        # Save cleaned images as cleaned instruction
        self.save_pdf(images=cleaned_images)

        return cleaned_images

    def save_pdf(self, images):
        # Create a new PDF document
        pdf_doc = fitz.open()

        # Iterate over the images
        for img in images:
            height, width, _ = img.shape
            img_rect = fitz.Rect(0, 0, width, height)
            page = pdf_doc.new_page(width=width, height=height)

            # Convert the NumPy array to a Pixmap object
            pixmap = fitz.Pixmap(fitz.csRGB, img.tobytes(), width, height)

            # Insert the image on the page
            page.insert_image(page.rect, pixmap=pixmap)

        # Save the PDF document
        pdf_doc.save("output.pdf")
