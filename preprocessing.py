import pandas as pd
import csv
import re
import numpy as np
from pyonmttok import Tokenizer

def clean_messages(messages):
    messages_cleaned = []
    for message in messages:
        message=message.lower()
        message = re.sub(r'(\w+:)*\/\S*', ' ', message)  # removes filepath/url
        message = re.sub(r'\S+\s=\s\S+', ' ', message)#string equality
        message = re.sub(r'\S+=(\w+)*', ' ', message)
        message = re.sub(r'(\w+-)*\w+((\.|:)\w+)*(\.|:)\w+', ' ', message) #removes no-textual strings with . : - inside 
        message = re.sub(r'\.\w+', ' ', message)#removes strings with .
        message = re.sub(r'\w+_\S+', ' ', message)#remove strings with underscore
        message = re.sub(r'(\S+-){2,}', ' ', message)#remove strings with more than two - 
        message = re.sub(r'(\w+\d+\w*)|(\d+\w+)', ' ', message)#remove alpha-numerical string
        message = re.sub(r'\[\w+\]', ' ', message)# removes [string]
        message = re.sub(r'(\d+)', ' ', message)#removes digits
        message = re.sub(r'[^\w\s]', ' ', message) # removes punctuation 
        message = re.sub(r'\\', ' ', message)#removes \
        #message = re.sub(r'\s([a-mp-z]){1,2}(\s|\Z)', ' ', message) #removes strings of one or two terms except for 'no'
        message = re.sub(r' +', r' ', message)# remove addictional whitespace  
        message = re.sub(r'\A ', '', message)#removes whitespace at the beginning
        message = re.sub(r' \Z', '', message)#removes whitespace at the end
        messages_cleaned.append(message)
  
    return messages_cleaned

def tokenize_message(message, tokenizer_type, spacer_annotate, preserve_placeholders,spacer_new):
    tokenizer = Tokenizer(tokenizer_type, spacer_annotate=spacer_annotate, preserve_placeholders= preserve_placeholders, spacer_new=spacer_new)
    return tokenizer.tokenize(message)[0]

             
def clean_tokenized(tokenized_message):
    """
    This function removes topwords
    """
           
    stoplist = ['the','with','a','an','but','of','on','to','all','has','have','been','for','in','it','its','itself',
                'this','that','those','these','is','are','were','was','be','being','having','had','does','did','doing',
                'and','if','about','again','then','so','too','cern','cms','atlas','by','srm','ifce', 'err']
    
    return [token for token in tokenized_message if token not in stoplist]
             

    
class MyCorpus(object):
    
    """An interator that yields sentences (lists of str)."""
    
    def __init__(self,inputDf):
        self.inputDf=inputDf
        self.list_err=self.inputDf["cleaned_strings"].tolist()      
    
    def __iter__(self):       
                     
        for line in self.list_err:
            tokenized=tokenize_message(line, 'space',False,True,False)
            yield clean_tokenized(tokenized)