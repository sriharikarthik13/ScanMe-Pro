import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy
import re
import string
import warnings
warnings.filterwarnings('ignore')

### Load NER model through spacy
model_ner = spacy.load('../model/output/model-best')

#A fucntion used to preprocess the text that is read through OCR methods and remove any form of punctuation
def cleanText(txt):
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans('','',whitespace)
    tablePunctuation = str.maketrans('','',punctuation)
    text = str(txt)
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)
    
    return str(removepunctuation)

# build a class that holds a function used to group the generated labels
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''
        
    def getgroup(self,text):
        if self.text == text:
            return self.id
        else:
            self.id +=1
            self.text = text
            return self.id
        

#Built a text parser using Regular Expressions, to retreive NER values of PHONE,EMAIL,WEB,ORG and incoporate with the already exsisting spacy NER to get a better and accurate output.
def parser(text,label):
    if label == 'PHONE':
        text = text.lower()
        text = re.sub(r'\D','',text)
        
    elif label == 'EMAIL':
        text = text.lower()
        allow_special_char = '@_.\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char),'',text)
        
    elif label == 'WEB':
        text = text.lower()
        allow_special_char = ':/.%#\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char),'',text)
        
    elif label in ('NAME', 'DES'):
        text = text.lower()
        text = re.sub(r'[^a-z ]','',text)
        text = text.title()
        
    elif label == 'ORG':
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]','',text)
        text = text.title()
        
    return text


#Calling the grouping class
grp_gen = groupgen()

#Building a function that processes the input data into entities that is used to feed the NER model
#Group the input text according to the NER labels, NAME,PHONE,EMAIL etc.
def getPredictions(image):
    # extract data using Pytesseract 
    tessData = pytesseract.image_to_data(image)
    # convert into dataframe
    tessList = list(map(lambda x:x.split('\t'), tessData.split('\n')))
    df = pd.DataFrame(tessList[1:],columns=tessList[0])
    df.dropna(inplace=True) # drop missing values
    df['text'] = df['text'].apply(cleanText)

    # convert data into content
    df_clean = df.query('text != "" ')
    content = " ".join([w for w in df_clean['text']])
    
    # get prediction from NER model
    doc = model_ner(content)

    # converting doc in json
    docjson = doc.to_json()
    doc_text = docjson['text']

    # creating tokens
    datafram_tokens = pd.DataFrame(docjson['tokens'])
    datafram_tokens['token'] = datafram_tokens[['start','end']].apply(
        lambda x:doc_text[x[0]:x[1]] , axis = 1)

    right_table = pd.DataFrame(docjson['ents'])[['start','label']]
    datafram_tokens = pd.merge(datafram_tokens,right_table,how='left',on='start')
    datafram_tokens.fillna('O',inplace=True)

    # join lable to df_clean dataframe
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x)+1).cumsum() - 1 
    df_clean['start'] = df_clean[['text','end']].apply(lambda x: x[1] - len(x[0]),axis=1)

    # inner join with start 
    dataframe_info = pd.merge(df_clean,datafram_tokens[['start','token','label']],how='inner',on='start')


    # Entities

    info_array = dataframe_info[['token','label']].values
    entities = dict(NAME=[],ORG=[],DES=[],PHONE=[],EMAIL=[],WEB=[])
    previous = 'O'

    for token, label in info_array:
        bio_tag = label[0]
        label_tag = label[2:]

        # step -1 parse the token
        text = parser(token,label_tag)

        if bio_tag in ('B','I'):

            if previous != label_tag:
                entities[label_tag].append(text)

            else:
                if bio_tag == "B":
                    entities[label_tag].append(text)

                else:
                    if label_tag in ("NAME",'ORG','DES'):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text

                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text



        previous = label_tag
        
    return  entities




#Double checking and correcting the missed results through Regular Expressions:


# Function to extract text from image using opencv
def extract_text_from_image(image_path):
   # Read the image using OpenCV
   image = cv2.imread(image_path)
   # Convert the image to gray scale
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   # Use pytesseract to extract text from the image
   text = pytesseract.image_to_string(gray)
   return text

# Function to classify text into name, phone, email, address, and designation using RE
def classify_text(text):
   lines = text.split('\n')
   name = None
   phone = None
   email = None
   designation = None
   # Regular expressions for phone and email
   phone_regex = re.compile(r'\+?\d[\d -]{8,}\d')
   email_regex = re.compile(r'\S+@\S+')
   # Common designations
   designation_keywords = [
       'Manager', 'Director', 'Engineer', 'Developer', 'Designer', 'Consultant',
       'Specialist', 'Coordinator', 'Administrator', 'Executive', 'Officer',
       'Lead', 'Head', 'Chief', 'Analyst', 'Supervisor', 'President', 'VP', 'Vice President','Intern','Architect','Data Analyst','Doctor'
   ]
   for line in lines:
       line = line.strip()
       if not line:
           continue
       # Check for email
       if email_regex.search(line):
           email = email_regex.search(line).group(0)
       # Check for phone number
       elif phone_regex.search(line):
           phone = phone_regex.search(line).group(0)
       # Check for designation keywords
       elif any(keyword.lower() in line.lower() for keyword in designation_keywords):
           designation = line
       # Assuming the name is the first non-empty line and doesn't contain numbers
       elif not any(char.isdigit() for char in line) and name is None:
           name = line
       # Remaining text could be address
   return {
       'Name': name,
       'Phone': phone,
       'Email': email,
       'Designation': designation,
   }

#Final function to pass the provided image through the NER model and also the RE function to generate outputs and store it in a list.
# This model is outputs the generated final => name, phone, email, designation
def extract_text(imgpath):
    
    imgh = cv2.imread(imgpath)

    evals = getPredictions(imgh)
    
    img_text = extract_text_from_image(imgpath)
    namecheck = classify_text(img_text)['Name']
    phonecheck = classify_text(img_text)['Phone']
    emailcheck = classify_text(img_text)['Email']
    designationcheck = classify_text(img_text)['Designation']
    namefinal= evals['NAME']
    if evals['NAME'] != []:
        for i in evals['NAME']:
            if i != '':
                namefinal= i
                
        
    phonefinal = evals['PHONE']
    if evals['PHONE'] != []:
        for i in evals['PHONE']:
            if i != '':
                phonefinal= i
                
    emailfinal = evals['EMAIL']
    if evals['EMAIL'] != []:
        for i in evals['EMAIL']:
            if i != '':
                emailfinal= i


    designationfinal = designationcheck
    if evals['NAME'] == []:
        if namecheck != "":
            namefinal = namecheck
    if evals['PHONE'] == []:
        if phonecheck != "":
            phonefinal = phonecheck
    if evals['EMAIL'] == []:
        if emailcheck != "":
            emailfinal = emailcheck


    listmain = [namefinal,phonefinal,emailfinal,designationfinal]
    return listmain


imgpath1 = './static/business_cards/business_card1.jfif'
print(extract_text(imgpath1))