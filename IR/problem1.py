# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:06:50 2018

@author: ejeev
"""

import os
import Stemmer
from nltk.corpus import stopwords
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
import numpy as np
import time
import csv
from math import*
import matplotlib.pyplot as plt 

#%%

path="C:\\Users\\ejeev\\Desktop\\IR\med 2\med 2\\MED.ALL"
with open(path,'r') as file:    
    data = file.read()

docs=data.split("\n.I")


def clean_text(doc):
    words=re.sub("[^a-zA-z0-9]"," ",doc)  # removing puncutations and otherchracters.  
    clean_words=words.lower().split()    # spliting docs into words
    stop=set(stopwords.words("english")) # stop words (NLTK module is used)
    stop.add('w')
    important_words= [w for w in clean_words if not w in stop] # removing stop words from list
    last_words=[Stemmer.PorterStemmer().stem(w,0,len(w)-1) for w in important_words] # stemming the words(given stemmer is used)
    
    return " ".join(last_words) # joining and returning 

cleaned_docs=[]
for doc in docs:
    cleaned_docs.append(clean_text(doc))
 
## saving docs to disk##
#with open("C:\\Users\\ejeev\\Desktop\\IR\med 2\med 2\\doc.txt", 'w',newline='') as myfile:
#    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#    wr.writerows(cleaned_docs)   

#%%
    
## creating inverted index matrix
def fn_tdm_df(docs, xColNames = None, **kwargs):
    ''' create a term document matrix as pandas DataFrame
    with **kwargs you can pass arguments of CountVectorizer
    if xColNames is given the dataframe gets columns Names'''

    #initialize the  vectorizer
    vectorizer = CountVectorizer(**kwargs)
    x1 = vectorizer.fit_transform(docs)
        #create dataFrame
    df = pd.DataFrame(x1.toarray().transpose(), index = vectorizer.get_feature_names())
    if xColNames is not None:
        df.columns = xColNames
            
    return df,vectorizer
            
    

#%%
# clacuating time taken to create tdm
start_time = time.time()
tdm,vec= fn_tdm_df(cleaned_docs)
print("--- %s seconds ---" % (time.time() - start_time))
collection_time=time.time()-start_time

## saving tdm (inverted index matrix to disk)### 
#tdm.to_csv("C:\\Users\\ejeev\\Desktop\\IR\med 2\med 2\datacsv")

#%%
## claculating max, min of posting list
le=[]
for i in range(tdm.shape[0]):
    le.append(np.count_nonzero(tdm[i:i+1]))

minposting_length=min(le)
maxposting_length=max(le)


## toltal words in the documets
total_Words=np.sum(np.sum(tdm[:10307]))

## calculating sizes of saved files
statinfo = os.stat("C:\\Users\\ejeev\\Desktop\\IR\med 2\med 2\datacsv")
size_of_tdm= statinfo.st_size

statinfodoc = os.stat("C:\\Users\\ejeev\\Desktop\\IR\med 2\med 2\doc.txt")
size_of_doc= statinfodoc.st_size


# relative sizes of files.
tdm_biggeras=size_of_tdm/size_of_doc

#%%

print("how many documents are in the collection :   "+str((len(docs))))
print("how many words are in the collection :   "+str(total_Words))
print("how many unique words are in the indexing vocabulary of the collection :   "+str((tdm.shape[0])))
print("How many postings (inverted entries) are created for this collection : "+str((tdm.shape[0])))
print("What is the length of the shortest and longest postings lists? Average length of the postingslists :")
print("longestposting lenght   :"+str(maxposting_length))
print("shortestposting lenght   :"+str(minposting_length))

print("                       ######Statistics######                     ")
print("                        Average words per documets:"+str(total_Words/len(docs)))
print("                        Average unique words per documets:"+str(tdm.shape[0]/len(docs)))
print("                        Average average lenght of posting lists:"+str(np.average(le)))
print("                        time taken to index collection  :"+str(collection_time)) 
print("                        size of the files(tdm+docs)   :"+str(size_of_tdm+size_of_doc)+" bytes")
print("                        inverted index "+str(tdm_biggeras)+" time bigger than document")


#%% Problem 2

path2="C:\\Users\\ejeev\\Desktop\\IR\med 2\med 2\\MED.QRY"
with open(path2,'r') as file:    
    quary = file.read()

QRY=quary.split("\n.I")

def clean_qry(doc):
    words=re.sub("[^a-zA-z0-9]"," ",doc)  # removing puncutations and otherchracters.  
    clean_words=words.lower().split()    # spliting docs into words
    stop=set(stopwords.words("english")) # stop words (NLTK module is used)
    stop.add('w')
    important_words= [w for w in clean_words if not w in stop] # removing stop words from list
    last_words=[Stemmer.PorterStemmer().stem(w,0,len(w)-1) for w in important_words] # stemming the words(given stemmer is used)
    
    return " ".join(last_words[1:len(last_words)]) # joining and returning 


Cleaned_QRY=[]
for qry in QRY:
    Cleaned_QRY.append(clean_qry(qry))
    
    
    
qry_rep=vec.transform(Cleaned_QRY).toarray() # change has been done in "fn_tdm_df to return vectorizer"

qry_rep_frame=pd.DataFrame(qry_rep.transpose(), index = vec.get_feature_names())



# Normalization is done for each document
for col in tdm:
    tdm[col]=tdm[col]-min(tdm[col])/(max(tdm[col])-min(tdm[col])) # this line will convert old "tdm" to normalized tdm
    
    

def cosinesimilarity(qry,doc):
    inner_prod =np.dot(qry,doc)
    norm_qry=np.linalg.norm(qry)
    norm_doc=np.linalg.norm(doc)
    return inner_prod/(norm_doc*norm_qry)

def jaccardsimilarity(qry,doc):
    intersection_cardinality = np.dot(qry,doc)
    qry_norm =np.sum(np.square(qry))
    doc_norm = np.sum(np.square(doc))    
    denominator =qry_norm+doc_norm -intersection_cardinality
    return intersection_cardinality/float(denominator)

# creating similarity score matrix
    
def rank_matrix(tdm,qry_rep_frame,similarity):
    rank_of_doc = np.zeros((qry_rep.shape[0],tdm.shape[1])) # initialize rank matrix with zero
    counter1=-1
    for i in tdm:
        counter1 +=1
        counter2= -1
        for j in qry_rep_frame:
            counter2 += 1
            a=similarity(qry_rep_frame[j],tdm[i])       # pair wise cosine similarity.
            rank_of_doc[counter2][counter1]=a
    return rank_of_doc

# funtion for ranking  
def ranks_sorces(data,top):
    indexs=[]
    value=[]
    for i in range(30):
        indexs.append(data.sort_values([i],ascending=False).head(top)[i].index.values)# retriving top n ranked documets for given quary .
        value.append(data.sort_values([i],ascending=False).head(top)[i].values) # Top n document's similarity scores
        
    return indexs,value

#%%
# jaccard similarity rank
    
start_time = time.time()
rank_of_doc_jacc =rank_matrix(tdm,qry_rep_frame,jaccardsimilarity)    #  matrix for jaccard similarity

rank_frame_jacc=pd.DataFrame(rank_of_doc_jacc.transpose())   # convert matix in to dataframe ( easy visuvalization )    




# jaccardsimilarity results 

Top_10_doc_jacc,Top_10_scores_jacc=ranks_sorces(rank_frame_jacc,10) #  top 10 results for each query with jaccard similarity

Best_rep_doc_jacc,Best_rep_scores_jacc=ranks_sorces(rank_frame_jacc,1) # Best document match for each quary with jaccard similarity
print("--- %s seconds --- via jaccard similarity" % (time.time() - start_time)) #time taken to retrive best documents for given quary via jaccard similarty
collection_time=time.time()-start_time


# cosine similarity rank
start_time = time.time()
rank_of_doc_cos=rank_matrix(tdm,qry_rep_frame,cosinesimilarity)  #  matrix for cisine similarity
rank_frame_cos=pd.DataFrame(rank_of_doc_cos.transpose())       # convert matix in to dataframe ( easy visuvalization ) 

# cosine similarity results
Top_10_doc_cos,Top_10_scores_cos=ranks_sorces(rank_frame_cos,10)     # top 10 results for each query with cosine similarity
Best_rep_doc_cos,Best_rep_scores_cos=ranks_sorces(rank_frame_cos,1)  # Best document match for each quary with cosine similarity
print("--- %s seconds --- via Cosine similarity" % (time.time() - start_time)) #time taken to retrive best documents for given quary via Cosine similarty
collection_time=time.time()-start_time


    
#%%

## deliverables


top_10_scores_cos_frame=pd.DataFrame(Top_10_scores_cos)
print("TOP 10 SCORES FOR EACH QUERIES WITH COSINE SIMILARITY")
print(top_10_scores_cos_frame)
top_10_doc_cos_frame=pd.DataFrame(Top_10_doc_cos)
print("TOP 10 DOC FOR EACH QUERIES WITH COSINE SIMILATIRY")
print(top_10_doc_cos_frame)

print(" ")
top_10_scores_jacc_frame=pd.DataFrame(Top_10_scores_jacc)
print("TOP 10 SCORES FOR EACH QUERIES WITH JACCARD SIMILARITY")
print(top_10_scores_jacc_frame)
top_10_doc_jacc_frame=pd.DataFrame(Top_10_doc_jacc)
print("TOP 10 DOC FOR EACH QUERIES WITH JACCARD SIMILATIRY")
print(top_10_doc_jacc_frame)

x = np.linspace(1, 30, 30) # query size 
# plotting the cosine similarity results
print("Query Vs Document")
print("")
plt.scatter(Best_rep_doc_cos,x, label = "cosine",marker='*',s=100) 
  


# plotting the jaccard similarity results
plt.scatter(Best_rep_doc_jacc,x, label = "jaccard", alpha=0.5) 
  
# naming the x axis 
plt.xlabel('Douments') 
# naming the y axis 
plt.ylabel('Quary') 
# giving a title to my graph 
plt.title('Best retrived Document for query') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 


print("Scores vs Documents")
print("")
plt.scatter(Best_rep_doc_cos,Best_rep_scores_cos, label = "cosine",marker='*',s=50) 
  
 

# plotting the jaccard similarity results
plt.scatter(Best_rep_doc_jacc,Best_rep_scores_jacc, label = "jaccard", alpha=0.5) 
  
# naming the x axis 
plt.xlabel('Douments') 
# naming the y axis 
plt.ylabel('Scores') 
# giving a title to my graph 
plt.title('Best retrived Document for query') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 


print("Scores Vs Queries")
print("")
plt.scatter(x,Best_rep_scores_cos, label = "cosine",marker='*',s=50) 
  


# plotting the jaccard similarity results
plt.scatter(x,Best_rep_scores_jacc, label = "jaccard", alpha=0.5) 
  
# naming the x axis 
plt.xlabel('Query') 
# naming the y axis 
plt.ylabel('Scores') 
# giving a title to my graph 
plt.title('Best retrived Document for query') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 


    
        