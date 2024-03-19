import requests
import os

from pypdf import PdfReader
from langchain_community.document_loaders import PDFMinerLoader


class Instructs:
    def __init__(self,
                 instr_dir,
                 pdf_path=None,
                 pdf_url=None):

        self.instr_dir = instr_dir

        assert pdf_path is not None or pdf_url is not None, "You must specify 'pdf_path' or 'pdf url'"
        self.instr_pdf = self.open_pdf(
            pdf_path) if pdf_path else self.download_pdf(pdf_url)

        self.__instr_imgs = None

    def open_pdf(self, pdf_path):
        instr_pdf = PdfReader(pdf_path)

        return instr_pdf

    def download_pdf(self, pdf_url):
        instr_name = os.path.basename(pdf_url)

        # Check if instruction was already downloaded
        if instr_name not in os.listdir(self.instr_dir):
            res = requests.get(pdf_url)
            pdf_path = os.path.join(self.instr_dir, instr_name)
            with open(pdf_path, 'wb') as f:
                f.write(res.content)
        else:
            pdf_path = os.path.join(self.instr_dir, instr_name)

        instr_pdf = self.open_pdf(pdf_path)

        return instr_pdf

    @property
    def instr_imgs(self):
        if self.__instr_imgs is None:
            self.__instr_imgs = self.extract_images()

        return self.__instr_imgs

    def extract_images(self):
        images = []
        for page in self.instr_pdf.pages:
            image = page.images[0].image
            images.append(image)

        return images


class InstructsOCR:
    def __init__(self):
        self.ocr_model = ''

    def parse_layout(self):
        pass

    def predict(self):
        pass
