# Question Answer App for Medicine Instructions

## Overview

The Question Answer App for Medicine Instructions is a comprehensive application designed to extract and analyze medical instruction documents to provide accurate and relevant answers to user queries. The app leverages advanced machine learning models and OCR technology to process and interpret complex medical texts, enabling users to search for specific instructions and obtain answers to their questions through a chat interface.

## Features

- **Layout Analysis with SegFormer**: Utilizes SegFormer, a state-of-the-art model for segmenting and understanding the layout of medical instruction documents.
- **Text Extraction with Tesseract**: Extracts text based on the analyzed layout using Tesseract OCR, ensuring accurate and efficient text recognition.
- **Question Answering with RAG (Retrieval Augmented Generation) and YandexGPT**: Implements a sophisticated RAG model in conjunction with YandexGPT to retrieve relevant document sections and generate precise answers to user questions.
- **Streamlit Interface**: A user-friendly Streamlit app that integrates search functionality and a chat interface for seamless interaction and query resolution.

## Components

1. **Layout Analysis Model (SegFormer)**:
   - Segments and analyzes the layout of the medical instruction documents to identify different sections and structures.
   
2. **Text Extraction (Tesseract)**:
   - Extracts text from the segmented document layout, ensuring high accuracy in recognizing and digitizing the content.
   
3. **Question Answering (RAG and YandexGPT)**:
   - Combines retrieval and generation capabilities to find relevant information in the extracted text and provide accurate answers to user queries. The Large Language Model (LLM) used in this project is YandexGPT, known for its robust language understanding capabilities.

4. **Streamlit Application**:
   - **Search Functionality**: Allows users to search for specific instructions within the documents.
   - **Chat Interface**: Enables users to ask questions and receive answers in a conversational manner.
