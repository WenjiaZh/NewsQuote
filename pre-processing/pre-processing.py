import json
import jsonlines
import basic
import pandas as pd
from sklearn.model_selection import train_test_split

file_path ='data path'

with open(file_path+'sentences_srl.json','r') as f:
    d = json.load(f)

# get id list
idlist = list(d.keys())

# read source data
n = 0
data_original = {}
with jsonlines.open('data path', mode='r') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        Id = str(obj['id'])
        data_original[Id]=obj
        n = n + 1   
        if n == 100000:
            break   

#read candidate trigger verbs
verb = pd.read_csv('SelectedTriggerVerbs.csv',header = None)
verblist = verb[0].to_list()


#Filter data by the trigger verbs and the srl labels
dataset =[]
cunt = 0
for k in range(len(idlist)):
    main_id = str(idlist[k])
    sub_id = 0
    obj = d[main_id]
    for i in range(len(obj)):
        sentence = obj[i]['verb_list']
        for j in range(len(sentence['verbs'])):
            if str(sentence['verbs'][j]['verb']) in verblist:
                if {'B-V','B-ARG1','B-ARG0'} < set(sentence['verbs'][j]['tags']):
                    Id = main_id+'-'+str(sub_id)
                    s = obj[i]['sent']
                    l = obj[i]['left']
                    r = obj[i]['right']
                    w = sentence['words']
                    t = sentence['verbs'][j]['tags']
                    verb = str(sentence['verbs'][j]['verb'])
                    dt = {}
                    dt['ID']=Id
                    dt['Left'] = l
                    dt['Sent']=s
                    dt['Right'] = r
                    dt['Tags']=t
                    dt['Words']=w
                    dt['Verb']=verb
                    dataset.append(dt)
                    sub_id = sub_id +1
    cunt = cunt+1
    if cunt%1000 == 0:
        #print(cunt/1000)
        with open(file_path+"verb_slect_sentence.json",'w') as f:
            json.dump(dataset,f)
         
with open(file_path+"verb_slect_sentence.json",'w') as f:
    json.dump(dataset,f)

print(len(dataset))    
#748216


#Enrich dataset with infromation from the source data

def tag_to_text(words,tags,target):
    text = []
    for i in range(len(words)):
        if tags[i] in target:
            text.append(words[i])
    return text

EntityTypes = pd.read_csv(file_path + 'SelectedOntologyClasses.txt',header = None)
entity_type = EntityTypes[0].to_list()
def read_entity(entity_list):
    #entities_text =[]
    entities = []
    for j in range(len(entity_list)):
        eb = entity_list[j]
        m = 0
        for k in range(len(eb['types'])):
            if eb['types'][k] in entity_type:
                m = m+1
                break
        if m != 0: 
            entities.append(eb)
            #entities_text.append(eb['text'])
    return entities #,entities_text


dataset_with_supplement = []
cunt = 0
for i in range(len(dataset)):
    instance = {} 
    instance['ID'] = dataset[i]['ID']
    instance['Sentence'] = dataset[i]['Sent']
    instance['Tags'] = dataset[i]['Tags']
    instance['Left_sentence'] = dataset[i]['Left']
    instance['Right_sentence'] = dataset[i]['Right']
    instance['Speaker'] = " ".join(tag_to_text(dataset[i]['Words'],dataset[i]['Tags'],['B-ARG0','I-ARG0']))
    instance['Words'] = dataset[i]['Words']
    instance['Verb'] = dataset[i]['Verb']
    instance['Quotation1'] = " ".join(tag_to_text(dataset[i]['Words'],dataset[i]['Tags'],['B-ARG1','I-ARG1']))
    instance['Quotation2'] = " ".join(tag_to_text(dataset[i]['Words'],dataset[i]['Tags'],['B-ARG2','I-ARG2']))
    instance['Title'] = data_original[str(dataset[i]['ID'][0:8])]['title']
    instance['Share'] = data_original[str(dataset[i]['ID'][0:8])]['social_shares_count']
    instance['Summary'] = data_original[str(dataset[i]['ID'][0:8])]['summary']['sentences']
    instance['Time'] = data_original[str(dataset[i]['ID'][0:8])]['published_at']
    instance['Category'] = basic.read_cat(data_original[str(dataset[i]['ID'][0:8])])
    instance['Entities'] = read_entity(data_original[str(dataset[i]['ID'][0:8])]['entities']['body'])
    instance['Hashtags'] = data_original[str(dataset[i]['ID'][0:8])]['hashtags']
    instance['Keywords'] = data_original[str(dataset[i]['ID'][0:8])]['keywords']
    instance['Source'] = data_original[str(dataset[i]['ID'][0:8])]['source']
    dataset_with_supplement.append(instance)
    #cunt = cunt+1
    #if cunt%1000 == 0:
        #print(cunt/1000)

with open(file_path+"dataset_with_supplement.json",'w') as f:
    json.dump(dataset_with_supplement,f)

print(len(dataset_with_supplement)) 
#748216     


#Filter data by the subject Ontology Classes
dataset_entity_in= []
for i in range(len(dataset_with_supplement)):
    obj = dataset_with_supplement[i]
    entity_list = obj['Entities']
    entities = []
    sequence = str(obj['Left_sentence'])+str(obj['Sentence'])+str(obj['Sentence'])
    for j in range(len(entity_list)):
        if entity_list[j]['text'] in sequence:
                entities.append(entity_list[j])
    if len(entities) > 0:
        obj['Entities_in'] = entities
        dataset_entity_in.append(obj)

with open(file_path+"dataset_entity_in.json",'w') as f:
    json.dump(dataset_entity_in,f)

print(len(dataset_entity_in)) 
#673701     


dataset_speaker_in = []
for i in range(len(dataset_entity_in)):
    obj = dataset_entity_in[i]
    entity_list = obj['Entities_in']
    e = []
    for j in range(len(entity_list)):
        e.append(entity_list[j]['text'])
    if obj['Speaker'] in e:
        for k in range(len(entity_list)):
            if obj['Speaker'] == entity_list[k]['text']:
                obj['Entity_Speaker'] = entity_list[k]
        dataset_speaker_in.append(obj)
       

with open(file_path+"dataset_speaker_in.json",'w') as f:
    json.dump(dataset_speaker_in,f)
print(len(dataset_speaker_in))   #149149


#Another dataset where the subject is a pronoun
pronoun = ['They','they','She','she','He','he','It','it','I']
dataset_pronoun_in = []
for i in range(len(dataset_entity_in)):
    obj = dataset_entity_in[i]
    if obj['Speaker'] in pronoun:
        dataset_pronoun_in.append(obj)

with open(file_path+"dataset_pronoun_in.json",'w') as f:
    json.dump(dataset_pronoun_in,f)
print(len(dataset_pronoun_in))   #135767  

#Filter the dataset by subject's DBpedia link
dataset_elink = []
for i in range(len(dataset_speaker_in)):
    obj = dataset_speaker_in[i]
    try:
        l =  obj['Entity_Speaker']['links']['dbpedia']
        obj['Entity_link'] = l
        dataset_elink.append(obj)
    except:
        continue

with open(file_path+"dataset_elink.json",'w') as f:
    json.dump(dataset_elink,f)
print(len(dataset_elink))           #35690


#We expect that the DBpedia link of the subject should appear at least twice.
e_links = {}
for i in range(len(dataset_elink)):
    l =  dataset_elink[i]['Entity_Speaker']['links']['dbpedia'] 
    if l not in e_links:
        e_links[l] = 1
    else:
        e_links[l] += 1
        
elink_list = list(e_links.keys())
e_filtered = {}
for elink in elink_list:
    if e_links[elink] > 1:
        e_filtered[elink] = e_links[elink]

elinks_morethan_2 = list(e_filtered.keys())

dataset_elinks_morethan2 = []

for i in range(len(dataset_elink)):
    l =  dataset_elink[i]['Entity_Speaker']['links']['dbpedia'] 
    if l in elinks_morethan_2:
        dataset_elinks_morethan2.append(dataset_elink[i])

with open(file_path + "dataset_elinks_morethan2.json",'w') as f:
    json.dump(dataset_elinks_morethan2,f)


#Filter the data by the length of quote and the number of quote marks
dataset_el_q1_morethan2 = []

for i in range(len(dataset_elinks_morethan2)):
    q1 =  dataset_elinks_morethan2[i]['Quotation1']
    q1_wordcount = len(q1.split())
    q_count =dataset_elinks_morethan2[i]['Sentence'].count('"')
    qq_count =dataset_elinks_morethan2[i]['Sentence'].count('”')
    if q1_wordcount > 2:
        if q_count % 2 == 0:
            if qq_count % 2 == 0:
                dataset_el_q1_morethan2.append(dataset_elinks_morethan2[i])

print('len(dataset_el_q1_morethan2)',len(dataset_el_q1_morethan2))
print(dataset_el_q1_morethan2[5].keys()) 

with open(file_path + "dataset_el_q1_morethan2.json",'w') as f:
    json.dump(dataset_el_q1_morethan2,f)
    
    
   
