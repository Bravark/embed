import openai
import json
import numpy as np
import textwrap
import re
from time import time,sleep
from decouple import config



#function to open file
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        content = infile.read()
        print(content)
        return content  
        
        

result = {'content':'example content'}

query = input("Enter your question here: ")

pp = open_file('prompt_answer.txt')

prompt = open_file('prompt_answer.txt').replace('<<PASSAGE>>', result['content']).replace('<<QUERY>>', query)

print('this is the promt',prompt)
print('this is the promt',pp)
