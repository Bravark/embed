import openai
import json
import numpy as np
import textwrap
import re
from time import time,sleep
from decouple import config




openai.api_key = config('OPENAI_API_KEY')

#function to open file
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


# function that takes a pieace of content and retrun a vector embedding of it
def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

# return dot product of two vectors which shows how similer the two vectors are
def similarity(v1, v2):  
    return np.dot(v1, v2)

# a function that take a text input and a vector input (index.json in this case) and return an ordered list of the scores of similarity in a "content : content , Score : Score format"
def search_index(text, data, count=5):
    #get the enbedding of the text
    vector = gpt3_embedding(text)
    #create an empty score list to put the content and it's score
    scores = list()
    # loop through the data(which usally a json file of vectors and it's contents, here it is generate from the make_index.py file), and the the similarity with the input text calculate the top count(number) score, order it and put it in a list. 
    for i in data:
        score = similarity(vector, i['vector'])
        #print(score)
        scores.append({'content': i['content'], 'score': score})
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    #print(ordered)
    return ordered[0:count]


def gpt3_completion(prompt, engine='text-davinci-002', temp=0.6, top_p=1.0, tokens=1000, freq_pen=0.25, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
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
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            with open('agpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI: ', oops)
            sleep(1)


if __name__ == '__main__':
    #open the file
    with open('index.json', 'r') as infile:
        data = json.load(infile)
    #print(data)
    
    while True:
        #take the user input
        query = input("Enter your question here: ")
        #print(query)
        # check the silimarity between the input and the indexed vector and reurn the score
        results = search_index(query, data)
        #print(results)
        #exit(0)
        answers = list()
        # answer the same question for all returned chunks
        for result in results:
            prompt = open_file('prompt_answer.txt').replace('<<PASSAGE>>', result['content']).replace('<<QUERY>>', query)
            #print('This is the prompt',prompt)
            answer = gpt3_completion(prompt)
            #print('\n\n', answer)
            answers.append(answer)
        # summarize the answers together
        all_answers = '\n\n'.join(answers)
        chunks = textwrap.wrap(all_answers, 4000)
        final = list()
        for chunk in chunks:
            prompt = open_file('prompt_summary.txt').replace('<<SUMMARY>>', chunk).replace('<<QUERY>>', query)
            summary = gpt3_completion(prompt)

            final.append(summary)
        
        print('\n\n=========\n\n', '\n\n'.join(final))