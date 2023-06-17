import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
import matplotlib.pyplot as plt
import seaborn as sns
file_path ='data path'


def data_statistics(dataname):
    with open(file_path+dataname+'.json','r') as f:
        data = json.load(f)
    print('Dataname:',dataname)
    print('Number of instances:', len(data))
    categories = {}
    for i in range(len(data)):
        catnames = data[i]['Category']
        for j in range(len(catnames)):
            if catnames[j] not in categories:
                categories[catnames[j]] = 1
            else:
                categories[catnames[j]] += 1
    categories_sort = list(sorted(categories.items(),key = lambda kv:(kv[1], kv[0]),reverse=True))
    print('Number of news categories:',len(categories_sort))
    print('Top 10 categories:')
    print(categories_sort[0:10])
    quote1_len = []
    quote1_wordscount = []
    for i in range(len(data)):
        q1 =  str(data[i]['Quotation1'])
        quote1_len.append(len(q1))
        quote1_wordscount.append(len(q1.split()))
    print('Quote1 length range:','[',min(quote1_len),',',max(quote1_len),']')
    print('Average quote1 length:', np.mean(quote1_len))
    print('Quote1 word count range:','[',min(quote1_wordscount),',',max(quote1_wordscount),']')
    print('Average quote1 word count:', np.mean(quote1_wordscount))   
    e_speaker= {}
    for i in range(len(data)):
        s = data[i]['Speaker']
        if s not in e_speaker:
            e_speaker[s] = 1
        else:
            e_speaker[s] += 1 

    es_sort = list(sorted(e_speaker.items(),key = lambda kv:(kv[1], kv[0]),reverse=True))
    print('Number of entity speakers:',len(es_sort))
    print('Top 10 speakers:')
    print(es_sort[0:10])
    e_links = {}
    for i in range(len(data)):
        try:
            l =  data[i]['Entity_Speaker']['links']['dbpedia'] 
            if l not in e_links:
                e_links[l] = 1
            else:
                e_links[l] += 1
        except:
            continue
    e_sort = list(sorted(e_links.items(),key = lambda kv:(kv[1], kv[0]),reverse=True))
    print('Number of spekaer entity DBpedia links:',len(e_sort))
    print('Top 10 links:')
    print(e_sort[0:10])

    
data_statistics('dataset_el_q1_morethan2')

############

def draw_distribution_histogram(nums,bins, xlabel, ylabel, is_hist=True, is_kde=False, is_rug=False, \
  is_vertical=False, is_norm_hist=False):
  sns.set() 
  sns.distplot(nums, bins=bins, hist=is_hist, kde=is_kde, rug=is_rug, \
    hist_kws={"color":"steelblue"}, kde_kws={"color":"purple"}, \
    vertical=is_vertical, norm_hist=is_norm_hist)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  plt.title("Distribution")
  plt.tight_layout()


with open(file_path+'dataname'+'.json','r') as f:
    dataset_el_q1_morethan2 = json.load(f)
quote1_len = []
quote1_wordscount = []
times = []
keywords = []
hashtags = []
sources = []
speakers = []

for i in range(len(dataset_el_q1_morethan2)):
    q1 =  str(dataset_el_q1_morethan2[i]['Quotation1'])
    quote1_len.append(len(q1))
    quote1_wordscount.append(len(q1.split()))
    times.append(dataset_el_q1_morethan2[i]['Time'])
    keywords.append(dataset_el_q1_morethan2[i]['Keywords'])
    hashtags.append(dataset_el_q1_morethan2[i]['Hashtags'])
    sources.append(dataset_el_q1_morethan2[i]['Source']['domain'])
    speakers.append(dataset_el_q1_morethan2[i]['Speaker'])

def generate_dict(l):
    d = {}
    for i in range(len(l)):
        item = l[i]
        if item not in d:
            d[item] = 1
        else:
            d[item] = d[item]+1
    return d

def unfoldlist(l):
    unfoldl = []
    for i in range(len(l)):
        try:
            item = list(set(l[i]))
            unfoldl = unfoldl + item
        except:
            continue
    return unfoldl

def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

quote1_len_dict = generate_dict(quote1_len)
quote1_wordscount_dict = generate_dict(quote1_wordscount)
times_dict = generate_dict(times)
sources_dict = generate_dict(sources)
speakers_dict = generate_dict(speakers)

k = unfoldlist(keywords)
h = unfoldlist(hashtags)
keywords_dict = generate_dict(k)
hashtags_dict = generate_dict(h)

#Quotation length distribution
plt.figure(figsize=(12, 8))
draw_distribution_histogram(quote1_len,50,'Quotetion1 length','Count')
plt.show()

plt.figure(figsize=(12, 8))
draw_distribution_histogram(quote1_len,50,'Quotetion1 length','Density',is_hist=True, is_kde=True, 
is_rug=False, is_vertical=False, is_norm_hist=False)
plt.show()

#Quotation word counts distribution
plt.figure(figsize=(12, 8))
draw_distribution_histogram(quote1_wordscount,50,'Quotetion1 words count','Count')
plt.show()

plt.figure(figsize=(12, 8))
draw_distribution_histogram(quote1_wordscount,50,'Quotetion1 words count','Density',is_hist=True, is_kde=True, 
is_rug=False, is_vertical=False, is_norm_hist=False)
plt.show()

#Date distribution
times_df = pd.DataFrame.from_dict(times_dict , orient='index',columns=['count'])
times_df['index'] = pd.to_datetime(times_df.index)
daily_times_df = times_df.resample('D',on='index')['count'].sum().to_frame()

plt.figure(figsize=(12, 8))
ax = sns.barplot(x = daily_times_df.index.strftime("%Y-%m-%d") ,y = 'count',data=daily_times_df, palette=colors_from_values(daily_times_df['count'], "YlOrRd"))
ax.bar_label(ax.containers[0]) 
plt.show() 

#Source distribution
sources_d = sorted(sources_dict.items(),key = lambda kv:(kv[1], kv[0]),reverse=True) 
sources_df = pd.DataFrame(sources_d ,columns=['source','count'])
sources_plot = sources_df[0:25]
plt.figure(figsize=(12, 8))
ax = sns.barplot(x = 'count',y = 'source',data=sources_plot , palette=colors_from_values(sources_plot['count'], "YlOrRd"),orient='h')
ax.bar_label(ax.containers[0]) 
plt.show() 


#Hashtag distribution
hashtags_d = sorted(hashtags_dict.items(),key = lambda kv:(kv[1], kv[0]),reverse=True) 
hashtags_df = pd.DataFrame(hashtags_d ,columns=['hashtags','count'])
hashtags_plot = hashtags_df[0:25]
plt.figure(figsize=(12, 8))
ax = sns.barplot(x = 'count',y = 'hashtags',data=hashtags_plot , palette=colors_from_values(hashtags_plot['count'], "YlOrRd"),orient='h')
ax.bar_label(ax.containers[0]) 
plt.show() 

#Keywords distribution
keywords_d = sorted(keywords_dict.items(),key = lambda kv:(kv[1], kv[0]),reverse=True) 
keywords_df = pd.DataFrame(keywords_d ,columns=['keywords','count'])
keywords_plot = keywords_df[0:25]
plt.figure(figsize=(12, 8))
ax = sns.barplot(x = 'count',y = 'keywords',data=keywords_plot , palette=colors_from_values(keywords_plot['count'], "YlOrRd"),orient='h')
ax.bar_label(ax.containers[0]) 
plt.show() 

#Category distribution
categories = {}
for i in range(len(dataset_el_q1_morethan2)):
    catnames = dataset_el_q1_morethan2[i]['Category']
    for j in range(len(catnames)):
        if catnames[j] not in categories:
            categories[catnames[j]] = 1
        else:
            categories[catnames[j]] += 1
category_d= sorted(categories.items(),key = lambda kv:(kv[1], kv[0]),reverse=True)

category_df = pd.DataFrame(category_d ,columns=['category','count'])
category_plot = category_df[0:25]
plt.figure(figsize=(12, 8))
ax = sns.barplot(x = 'count',y = 'category',data=category_plot , palette=colors_from_values(category_plot['count'], "YlOrRd"),orient='h')
ax.bar_label(ax.containers[0]) 
plt.show() 

#Speakers distribution
speakers_d = sorted(speakers_dict.items(),key = lambda kv:(kv[1], kv[0]),reverse=True) 
speakers_df = pd.DataFrame(speakers_d ,columns=['speakers','count'])
speakers_plot = speakers_df[0:25]
plt.figure(figsize=(12, 8))
ax = sns.barplot(x = 'count',y = 'speakers',data=speakers_plot , palette=colors_from_values(speakers_plot['count'], "YlOrRd"),orient='h')
ax.bar_label(ax.containers[0]) 
plt.show() 



