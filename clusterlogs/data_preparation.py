import re
import pprint

# def clean_messages(messages):
#     messages_cleaned = []
#     for item in messages:
#         # print(item)
#         item = re.sub(r'\S+\.\S+', ' ', item)  # any URL
#         # item = re.sub(r'(/[\w\./]*\s?)', ' ', item)
#         # item = re.sub(r'([a-zA-Z0-9]+[_]+\S+)', ' ', item)
#         item = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', ' ', item) # remove all substrings with digits
#         item = re.sub(r'(\d+)', ' ', item) # remove all other digits
#         item = re.sub(r'[^\w\s]', ' ', item) # removes all punctuation
#         item = re.sub(r' +', r' ', item)
#         messages_cleaned.append(item.lower())
#         # print(item)[A-Z_]{2,}
#     # pprint.pprint(messages_cleaned)
#     return messages_cleaned
def pre_cleaning(messages):
    messages_cleaned=[]
    for message in messages:
        message=re.sub(r'[A-Z_]{2,}', '', message) #removes upper case strings \w+::\S+
        message=re.sub(r'\w+::\S+', '', message)
        message=re.sub(r'(?i)(copy.+failure:)', '', message)
        message=re.sub(r'(?i)(commu.+err:)', '', message)
        message=re.sub(r'(?i)(globus.+\d+\s\d+-)', '', message)
        message=re.sub(r'(?i)(ser.+res.+ror)', '', message)
        #message = re.sub(r' +', r' ', message)# remove addictional whitespace  
        message = re.sub(r'\A ', '', message)#removes whitespace at the beginning
        message = re.sub(r' \Z', '', message)#removes whitespace at the end
        messages_cleaned.append(message)  
    return messages_cleaned

def clean_messages(messages):
    messages_cleaned = []
    for message in messages:
        message=message.lower()
        message = re.sub(r'(\w+:)*\/\S*', ' ', message)  # removes filepath/url
        message = re.sub(r'\S+\s=\s\S+', ' ', message)#string equality
        message = re.sub(r'\S+=(\w+)*', ' ', message)

        message = re.sub(r'(\w+-)*\w+((\.|:)\w+)*(\.|:)\w+', ' ', message) #removes no-textual strings with . : - inside 

        message = re.sub(r'\.\w+', ' ', message)#removes strings with .
        #message = re.sub(r'(\w+\.)+\w+ ', ' ', message)
        message = re.sub(r'\w+_\S+', ' ', message)#remove strings with underscore
        message = re.sub(r'(\S+-){2,}', ' ', message)#remove strings with more than two - 
        message = re.sub(r'(\w+\d+\w*)|(\d+\w+)', ' ', message)#remove alpha-numerical string

        #message = re.sub(r'(o=)(\w+\s)+\w+', 'o=', message) 
        #message = re.sub(r'(ou=)(\w+\s)+\w+', 'ou=', message) 
        #message = re.sub(r'=\w+','=', message)
        message = re.sub(r'\[\w+\]', ' ', message)# removes [string]
        message = re.sub(r'(\d+)', ' ', message)#removes digits
        message = re.sub(r'[^\w\s]', ' ', message) # removes punctuation 
        message = re.sub(r'\\', ' ', message)#removes \
        message = re.sub(r'\s([a-mp-z]){1,2}(\s|\Z)', ' ', message) #removes strings of one or two terms except for 'no'
        
        #message = re.sub(r'\s=\s', ' ', message)
        message = re.sub(r' +', r' ', message)# remove addictional whitespace  
        message = re.sub(r'\A ', '', message)#removes whitespace at the beginning
        message = re.sub(r' \Z', '', message)#removes whitespace at the end
        messages_cleaned.append(message)
  
    return messages_cleaned


def alpha_cleaning(messages):
    messages_cleaned = []
    for item in messages:
        item = re.sub(r'\S+\.\S+', ' ', item)  # any URL
        item = re.sub(r'(\d+)', ' ', item)
        item = re.sub(r'[^\w\s]', ' ', item)
        item = re.sub(r' +', r' ', item)
        messages_cleaned.append(item)
    return messages_cleaned