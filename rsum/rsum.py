import openai
import os
from time import time,sleep
import textwrap
import re
from decouple import config
import pdfplumber
import numpy as np
            
openai.api_key = config('OPENAI_API_KEY')

# function to read the file from a directory and extract the text.
def convert_pdf2txt(src_dir):
    files = os.listdir(src_dir)
    files = [i for i in files if '.pdf' in i]
    for file in files:
        try:
            with pdfplumber.open(src_dir+file) as pdf:
                output = ''
                for page in pdf.pages:
                    output += page.extract_text()
                    output += '\n\nNEW PAGE\n\n'  # change this for your page demarcation
                #print(output)
                pdf2text = output  
                return pdf2text 
               # save_file(dest_dir+file.replace('.pdf','.txt'), output.strip())
        except Exception as oops:
            print(oops, file)


#function to open a file
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

#function to save file
def save_file(content, filepath):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.6, top_p=1.0, tokens=2000, freq_pen=0.25, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            #remove too many new lines
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)
            
            
            
            
   

    
    


if __name__ == '__main__':
    #open the file that has the input
    alltext = convert_pdf2txt('PDFs/')
    #use text wrap to cut the input into chucks 
    chunks = textwrap.wrap(alltext, 3500)
    #create amd empty list to put the results
    result = list()
    # count to see how may summery is being generated
    count = 0
    for chunk in chunks:
        #increament count
        count = count + 1
        # open the file that contains the Prompt format and replace the place-holder for the text to summerize with the chucks from the input.
        prompt = open_file('prompt.txt').replace('<<SUMMARY>>', chunk)
        #encode the prompt and decode (still don't know why they dp this)
        prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
        # send to GPT
        summary = gpt3_completion(prompt)
        #format the response text in triple new line and add the count
        print('\n\n\n', count, 'of', len(chunks), ' - ', summary)
        #add it to the result list
        result.append(summary)
    # save the file    
    save_file('\n\n'.join(result), 'output_%s.txt' % time())