from dataclasses import dataclass, field
import os
import re
import shutil
import tkinter as tk
from tkinter import messagebox
import os
from tkinter import simpledialog
from tkinter import filedialog
import sys
import configparser
import sys
from PyPDF2 import PdfWriter, PdfReader
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import fitz
import numpy as np
import cv2

def separatePages(pdfPath):
    

    stringsToExclude = ["GLASS ORDER", "TEMPLATE"]

    pdf = PdfReader(pdfPath)
    # Loop through each page of the existing PDF
    for pageNum in range(len(pdf.pages)):
        
        output = PdfWriter()
        currentPage = pdf.pages[pageNum]
        pageText = currentPage.extract_text().upper()
        if any(s in pageText for s in stringsToExclude):
            continue

        output.add_page(currentPage)
        fileName, ext = os.path.splitext(pdfPath)
        newFileName = fileName + "_" + str(pageNum) + ext
        # Write the output to a new PDF file
        outputStream = open(newFileName, "wb")
        output.write(outputStream)
        outputStream.close()
    
    os.remove(pdfPath)
    
def exportImage(pdf_path, page_number):
    pdf_document = fitz.open(pdf_path)
    
    page = pdf_document.load_page(page_number)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
    
    # Get image data as bytes
    image_data = pixmap.samples

    # Convert bytes to a NumPy array
    image_array = np.frombuffer(image_data, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)

    pdf_document.close()
    
    return image_array

dir = r'C:\Users\alaci\OneDrive - Arrow Glass Industries\Documents\Scripts\Test\imgRec\trainingData\doorType'
for pdfFile in os.listdir(dir):
    if pdfFile.lower().endswith(".pdf"):
        pdfFile = os.path.join(dir,pdfFile)
        separatePages(pdfFile)

for pdfFile in os.listdir(dir):
    if pdfFile.lower().endswith(".pdf"):
        pdfFile = os.path.join(dir,pdfFile)
        image = exportImage(pdfFile,0)
        imageName = pdfFile.replace(".pdf",".jpg")
        cv2.imwrite(imageName,image)
        os.remove(pdfFile)
        
