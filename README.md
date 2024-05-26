# Question Answer App for Medicine Instructions

## Overview

The Question Answer App for Medicine Instructions is a comprehensive application designed to extract and analyze medical instruction documents to provide accurate and relevant answers to user queries. The app leverages advanced machine learning models and OCR technology to process and interpret complex medical texts, enabling users to search for specific instructions and obtain answers to their questions through a chat interface.

## Components

1. **Stamp Detection and Removal with YOLO v8**: Uses the YOLO v8 model to detect and remove stamps from documents, improving the clarity and quality of the extracted text.

2. **Layout Analysis with SegFormer**: Utilizes finetuned SegFormer model to segment and understand the layout of medical instruction documents.

3. **Text Extraction with Tesseract**: Ensures accurate text recognition by extracting text based on the analyzed layout using Tesseract OCR.

4. **Question Answering with RAG and FAISS/OpenSearch**: Implements a Retrieval Augmented Generation (RAG) model in conjunction with FAISS or OpenSearch for efficient and effective retrieval of relevant document sections.

5. **Answer Generation with YandexGPT**: Employs YandexGPT, a robust LLM known for its Russian language understanding capabilities, to generate answers to user questions.

6. **Streamlit Interface**: Offers a user-friendly interface that integrates search functionality and a chat interface for seamless interaction and query resolution.
