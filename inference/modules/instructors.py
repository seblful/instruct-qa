import os
import re
import requests
import io
from PIL import Image

from pypdf import PdfReader
import fitz


class Instruction:
    def __init__(self,
                 instr_dir,
                 pdf_path=None,
                 pdf_url=None):

        self.instr_dir = instr_dir

        self.pdf_path = pdf_path
        self.pdf_url = pdf_url

        self.url_regexp = re.compile(
            r"(https?://)?(www\.)?rceth\.by\/NDfiles\/instr\/[0-9_]*_[isp]\.pdf")

        # Assert errir if pdf_path and pdf_url is None
        assert pdf_path is not None or pdf_url is not None, "You must specify 'pdf_path' or 'pdf url'"
        self.instr_pdf = self.open_pdf(
            pdf_path) if pdf_path else self.download_pdf(pdf_url)

        self.__instr_imgs = None

    def open_pdf(self, pdf_path):
        instr_pdf = fitz.open(pdf_path)

        return instr_pdf

    def download_pdf(self, pdf_url):
        # Check if url is valid
        if not self.validate_url(url=pdf_url):
            raise ValueError('Url is not valid')

        # Retrieve name of file
        instr_name = os.path.basename(pdf_url)

        # Check if instruction was already downloaded
        if instr_name not in os.listdir(self.instr_dir):
            res = requests.get(pdf_url)
            pdf_path = os.path.join(self.instr_dir, instr_name)
            with open(pdf_path, 'wb') as f:
                f.write(res.content)
        else:
            pdf_path = os.path.join(self.instr_dir, instr_name)

        # Set pdf path to instance
        self.pdf_path = pdf_path

        # Open instruction
        instr_pdf = self.open_pdf(pdf_path)

        return instr_pdf

    def validate_url(self, url):
        result = self.url_regexp.match(url)

        if result:
            return True

        return False

    @property
    def instr_imgs(self):
        if self.__instr_imgs is None:
            self.__instr_imgs = self.extract_images()

        return self.__instr_imgs

    def extract_images(self):
        # Create empty list to store images
        images = []
        # Iterate through all the pages of the PDF
        for page_index in range(self.instr_pdf.page_count):
            # Get the current page
            page = self.instr_pdf[page_index]

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

                if image_extension != 'jpeg':
                    image = image.convert('RGB')

                # Append image to images list
                images.append(image)

        return images
