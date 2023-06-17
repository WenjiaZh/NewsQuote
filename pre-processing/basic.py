import re
from matplotlib.pyplot import title
import unidecode
import pandas as pd
import numpy as np
import jsonlines
import json
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import default_data_collator
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import collections
from tqdm.auto import tqdm

codepath = 'your path'
IPTCcode = pd.read_excel(codepath+'IPTC-Subject-NewsCodes.xls')
def findIPTC (iptccode):
    iptccode = 'subj:'+ iptccode
    index = IPTCcode[(IPTCcode.NewsCode == iptccode)].index.tolist()
    if len(index) == 1 :
        newsname = IPTCcode.iloc[index[0]][1]
        newsdef = IPTCcode.iloc[index[0]][2]
        return newsname, newsdef 
    else:
        return 'Error','Error'



IABcode = np.load(codepath+'IABcategories.npy',allow_pickle='TRUE').item()
def findIAB (iabcode):
        return IABcode[iabcode]

def readdata(path,num):
    data =[]
    n=0
    with jsonlines.open(path, mode='r') as reader:
        for row in reader:
         data.append(row)  
         n = n+1
         if n == num:
               break
    return data

def clean_text(sentence):
    sentence = re.sub(r"((https?|ftp)://)?[a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.[a-z]{2,}[a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*", "", sentence)
    sentence = re.sub(r"\s", " ", sentence)
    sentence = re.sub(r"(\.|!|\?) ", "", sentence) #or use EOS
    sentence = re.sub(r"[^A-Za-z0-9 ]", "", sentence)
    sentence = re.sub(r"\b[0-9]+\b", "",sentence)
    sentence = re.sub(r"\b[a-z0-9]\b", "", sentence)
    sentence = ' '.join(sentence.split())
    if sentence.startswith(" "):
        sentence = sentence[1:len(sentence)]

    return sentence

def process_one(data):
    paragraphs = []
    person_names = []
    ids =[]
    for i in range(len(data)):
        object = data[i]
        body = object['body']
        pn = object['paragraphs_count']
        bodies = [unidecode.unidecode(item) for item in body.split('\n',pn) if '"' in item]
        entities = object['entities']['body'] 
        person_list = [item for item in entities if 'Person' in item['types']]
        name_list =[item['text'] for item in person_list]
        bodies_plus = [item for item in bodies if any(i in item for i in name_list)]
        for j in range(len(bodies_plus)):
            id = str(object['id'])+'-'+str(j)
            paragraphs.append(bodies_plus[j])
            person_names.append(name_list)
            ids.append(id)
    datalist = zip(paragraphs,person_names,ids)
    namelist = ['Bodytext','Names','ID']
    df = pd.DataFrame(datalist, columns = namelist)
    datalist = zip(paragraphs,person_names,ids)
    namelist = ['Bodytext','Names','ID']
    df = pd.DataFrame(datalist, columns = namelist)
    return df

def read_cat(object):
    c = []
    category = object['categories']
    for i in range(len(category)):
        cat = category[i]
        if cat['taxonomy'] == 'iptc-subjectcode':
            catname, catdef = findIPTC(str(cat['id']))
        else:
            catname = findIAB(str(cat['id']))
        c.append(catname)
    return c
            


def process_QA(object,name_list):
    paragraphs = []
    person_names = []
    ids =[]
    category =[]
    time = []
    summary = []
    title = []
    share = []
    body = object['body']
    pn = object['paragraphs_count']
    t = object['published_at']
    s = object['summary']
    ti = object['title']
    sc = object['social_shares_count']
    cat = read_cat(object)
    bodies = [unidecode.unidecode(item) for item in body.split('\n',pn) if '"' in item]
    bodies_plus = [item for item in bodies if any(i in item for i in name_list)]
    for j in range(len(bodies_plus)):
        id = str(object['id'])+'-'+str(j)
        paragraphs.append(bodies_plus[j])
        person_names.append(name_list)
        ids.append(id)
        category.append(cat)
        time.append(t)
        summary.append(s)
        title.append(ti)
        share.append(sc)
    datalist = zip(paragraphs,person_names,ids,category,time,summary,title,share)
    namelist = ['Bodytext','Names','ID','Category','Time','Summary','Title','SocialShare']
    df = pd.DataFrame(datalist, columns = namelist)
    return df


def process_QA_(object,name_list):
    paragraphs = []
    person_names = []
    ids =[]
    category =[]
    time = []
    summary = []
    title = []
    share = []
    body = object['body']
    pn = object['paragraphs_count']
    t = object['published_at']
    s = object['summary']
    ti = object['title']
    sc = object['social_shares_count']
    cat = read_cat(object)
    bodies = [unidecode.unidecode(item) for item in body.split('\n',pn)]
    bodies_plus = [item for item in bodies if any(i in item for i in name_list)]
    for j in range(len(bodies_plus)):
        id = str(object['id'])+'-'+str(j)
        paragraphs.append(bodies_plus[j])
        person_names.append(name_list)
        ids.append(id)
        category.append(cat)
        time.append(t)
        summary.append(s)
        title.append(ti)
        share.append(sc)
    datalist = zip(paragraphs,person_names,ids,category,time,summary,title,share)
    namelist = ['Bodytext','Names','ID','Category','Time','Summary','Title','SocialShare']
    df = pd.DataFrame(datalist, columns = namelist)
    return df

def cleantext(sentence):
    r='[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    sentence = re.sub(r,'',sentence)
    sentence = re.sub(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "", sentence)
    sentence = re.sub(r"\t", " ", sentence)
    sentence = re.sub(r"\r", " ", sentence)
    return sentence


def extracleantext(sentence):
    sentence = re.sub(r'\n', " ", sentence)
    sentence = re.sub(r'\\', "\\\\", sentence)
    sentence = re.sub(r'\xad', " ", sentence)
    sentence = re.sub(r'\x93', " ", sentence)
    sentence = re.sub(r'\x94', " ", sentence)
    return sentence    

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
def process_tc(data):
    d = []
    for i in range(len(data)):
        ids = data.iloc[i,2]    
        sentence = str(clean_text(data.iloc[i,0]))
        tokens = tokenizer.tokenize(sentence) 
        tags = ['c']*len(tokens)
        row = [ids,tokens,tags]
        d.append (row)
    d = pd.DataFrame(d)
    d.rename(columns={0:'quoteID',1:'text',2:'tags'},inplace=True)
    return d

class NewDataset(Dataset):
    def __init__(self,ids,labels):
        self.ids = ids
        self.labels = labels
        self.len = len(ids)

    def __getitem__(self, item):
        tokens_tensor = torch.tensor(self.ids[item])
        label_tensor = torch.tensor(self.labels[item])
        return (tokens_tensor,label_tensor)

    def __len__(self):
        return self.len

def token_to_ids(tokenlist):
    ids = []
    for text in tokenlist:
        ids.append(tokenizer.convert_tokens_to_ids(text))
    return ids

MaxLen = 268
 
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    label_tensors = [s[1] for s in samples]
    one = [0]
    tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)
    label_tensors = pad_sequence(label_tensors,batch_first=True,padding_value=0)

    if len(tokens_tensors[0]) != MaxLen:
      tokens_tensors = torch.tensor([t + one for t in tokens_tensors.numpy().tolist()])
    if len(label_tensors[0]) != MaxLen: 
      label_tensors = torch.tensor([t + one for t in label_tensors.numpy().tolist()])
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, masks_tensors, label_tensors

def find_quo (news):    
    key = '"'
    countStr = news.count(key)
    quotes = []
    if (countStr  % 2) == 0:
        str = news
        for j in range(int(countStr/2)):
            index_start = str.find(key)
            str_new = str[index_start+1:len(str)+1]
            index_end = str_new.find(key)
            quo = str[index_start+1:index_start+index_end+1]
            quotes.append(quo)
            str = str_new[index_end+1:len(str)+1]
    return quotes      


def cut_sentences(content):
    content = content.replace('\n','')
    list_ret = re.split(r'(\.|\!|\?)', content)
    ret_list =[s for s in list_ret if len(s)!=0] 
    ret = []
    for i in range(len(ret_list)):
        if ret_list[i] not in ['.','!','?']:
            start_pos = i
            for j in range(i+1,len(ret_list)):
                if ret_list[j] not in ['.','!','?']:
                    end_pos = j
                    break
            if end_pos == start_pos:
                end_pos = len(ret_list)
            s = ''
            for n in range(start_pos,end_pos):
                s = s + ret_list[n]
            ret.append(s)
    for i in range(1,len(ret)):
        ret[i]=ret[i][1:len(ret[i])]
    return(ret) 


def find_all(s, c):
    pos = []
    idx = int(s.find(c))
    pos.append(idx)
    while idx != -1:
        yield idx
        idx = int(s.find(c, idx + 1))
        pos.append(idx)
    return pos


def find_first(s, c):
    idx = int(s.find(c))
    answer = c
    if idx == -1:
        speakers = c.split()
        wordlist=[]
        for name in speakers:
            idx_ = int(s.find(name))
            if idx_ != -1:
                words = {'word':name,'start_pos':idx_}
                wordlist.append(words)
        wordlist = sorted(wordlist, key=lambda x: x['start_pos'])
        if len(wordlist) != 0:
            idx = wordlist[0]['start_pos']
            answer = wordlist[0]['word']
        else:
            idx = 0
            answer = 'None' 
    return idx, answer



def find_longest(s, c):
    idx = int(s.find(c))
    if idx == -1:
        longest = max(c.split(), key=len, default='')
        idx = int(s.find(longest))
    return idx



def find_remove_first(s, c):
    idx = int(s.find(c))
    names = c
    answer = c
    while idx ==-1:
        names = ' '.join(names.split()[1:])
        #print(names)
        idx = int(s.find(names))
        answer = names
        if len(names) == 0:
            idx = 0
            answer = 'None'
            break
    return idx, answer

def find_remove_last(s, c):
    idx = int(s.find(c))
    names = c
    answer = c
    while idx ==-1:
        names = ' '.join(names.split()[:-1])
        #print(names)
        idx = int(s.find(names))
        answer = names
        if len(names) == 0:
            idx = 0
            answer = 'None'
            break
    return idx, answer

def find_answer_pos(s,c):
    answer = c
    idx = int(s.find(c))
    if idx == -1:
        idx, answer = find_remove_first(s,c)
        if idx == 0: 
            idx, answer = find_remove_last(s,c)
            if idx == 0:
                idx, answer = find_first(s,c)

    return idx, answer



