
import openai
import json
import textwrap
from decouple import config
import os
import pdfplumber
import re
import pandas as pd
import PyPDF4




openai.api_key = config('OPENAI_API_KEY')

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)           


def convert_pdf2txt(src_dir,):
    print("converting pdf to text")
    files = os.listdir(src_dir)
    files = [i for i in files if '.pdf' in i]
    for file in files:
        try:
            with pdfplumber.open(src_dir+file) as pdf:
                print("converting ",file,"to text")
                output = ''
                for page in pdf.pages:
                    output += page.extract_text()
                    output += ''  # change this for your page demarcation
                save_file(file.replace('.pdf','.txt'), output.strip())
                return output.strip()
        except Exception as oops:
            print(oops, file)
            
            

# def extract_pdf_text(file_path):
#     with open(file_path, 'rb') as pdf_file:
#         pdf_reader = PyPDF4.PdfFileReader(pdf_file)
#         text = ''
#         for page in range(pdf_reader.getNumPages()):
#             page_obj = pdf_reader.getPage(page)
#             page_text = page_obj.extractText()
#             text += page_text
#         return text

# def extract_headings_and_paragraphs(pdf_text):
#     lines = pdf_text.split('\n')
#     headings = []
#     paragraphs = []
#     current_heading = ''
#     current_paragraph = ''
#     current_font_size = -1
#     for line in lines:
#         if line.strip() == '':
#             continue
#         try:
#             font_size = float(line.split(' ')[-2])
#         except:
#             font_size = -1
#         if font_size > current_font_size:
#             if current_heading != '':
#                 headings.append(current_heading)
#                 paragraphs.append(current_paragraph)
#             current_heading = line
#             current_paragraph = ''
#             current_font_size = font_size
#         else:
#             current_paragraph += line + '\n'
#     headings.append(current_heading)
#     paragraphs.append(current_paragraph)
#     return pd.DataFrame({'Heading': headings, 'Paragraph': paragraphs})


            


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


if __name__ == '__main__':
    alltext = convert_pdf2txt('PDFs/')
    
    # dftext = extract_pdf_text('PDFs\MME 331 Lecture Notes_Smelting and Refining.pdf')
    # ddd = extract_headings_and_paragraphs(dftext)
    
    #alltext = open_file('input.txt')
    chunks = textwrap.wrap(alltext, 2500)
    print(chunks[0])
    # parts = extract_paragraphs(alltext)
    result = list()
    # print(ddd)
    for chunk in chunks:
        embedding = gpt3_embedding(chunk.encode(encoding='ASCII',errors='ignore').decode())
        info = {'content': chunk, 'vector': embedding}
        print(info, '\n\n\n')
        result.append(info)
    with open('index.json', 'w') as outfile:
        json.dump(result, outfile, indent=2)