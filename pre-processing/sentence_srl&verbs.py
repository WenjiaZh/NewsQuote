import jsonlines
import json
import pandas as pd
import re
from allennlp.predictors.predictor import Predictor

print('start')
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

print(predictor)


file_path = 'data path'
save_path = 'output path'

def cut_sentences(content):
    content = content.replace('\n',' ')
    list_ret = re.split(r'(\.|\!|\?)', content)
    ret_list =[s for s in list_ret if len(s)!=0 ] 
    ret = []
    for i in range(len(ret_list)):
        if ret_list[i] not in ['.','!','?']:
            start_pos = i
            end_pos = len(ret_list)
            for j in range(i+1,len(ret_list)):
                if ret_list[j] not in ['.','!','?']:
                    end_pos = j
                    break
            s = ''
            for n in range(start_pos,end_pos):
                s = s + ret_list[n]
            ret.append(s)
    for i in range(1,len(ret)):
        ret[i]=ret[i][1:len(ret[i])]
    return(ret) 


print('start processing')

n = 0
D = {}
verb_D = {}
with jsonlines.open(file_path, mode='r') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        ids = obj['id']
        body = obj['body']
        sentences = cut_sentences(body)
        s_list = []
        for i in range(len(sentences)):
            s_instance = {}
            if i == 0:
                s_instance['left'] = []
            else:
                s_instance['left'] = sentences[i-1]
            if i == len(sentences)-1:
                s_instance['right'] = []
            else:
                s_instance['right'] = sentences[i+1]
            s_instance['sent'] = sentences[i]
            try:
                s_pred = predictor.predict(sentence=s_instance['sent'])
                s_instance['verb_list'] = s_pred
                s_list.append(s_instance)
                for j in range(len(s_pred['verbs'])):
                    verb = s_pred['verbs'][j]['verb']
                    if verb not in verb_D:
                        verb_D[verb] = 1
                    else:
                        verb_D[verb] += 1 
            except:
                continue
        D[str(ids)] = s_list
        n = n + 1   
        if n%200 == 0:
            print(n/100000,'%')
            with open(save_path+'sentences_srl.json','w') as f:
                json.dump(D,f) 
            with open(save_path+'verbs.json','w') as f:
                json.dump(verb_D,f) 
        if n == 100000: #Only use the first 100000 articles
            break    

with open(save_path+'sentences_srl.json','w') as f:
    json.dump(D,f) 
with open(save_path+'verbs.json','w') as f:
    json.dump(verb_D,f) 

print('Finish')

print('generate excel')

with open(save_path+'verbs.json','r') as f:
    verb_dict = json.load(f)

verb_dict = sorted(verb_dict.items(),key = lambda kv:(kv[1], kv[0]),reverse=True) 
verb_df = pd.DataFrame(list(verb_dict))
verb_df.columns = ['Verb','Count']
#verb_df.to_csv(save_path+"verb.csv",index=False)
verb_df.to_excel(save_path+'verb.xlsx', encoding='utf-8', index=False)
print('Finish All')
