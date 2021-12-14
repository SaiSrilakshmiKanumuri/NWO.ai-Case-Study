#!/usr/bin/env python
# coding: utf-8

# # Challenge: Semantic Search Algorithm
# Design and implement a semantic search algorithm that is able to score and rank a
# set of keywords (trends) by how strongly associated they are to a given query term. A
# simple approach could borrow techniques from association rule mining to analyze the
# co-occurrence of terms within a corpora of tweets and reddit posts -- other
# approaches are also welcome. Regardless of the methodology, the solution should
# take into consideration the uniqueness of the trend and the recency of the
# association. For example, the algorithm should be able to determine that the query
# ‘iPhone’ is more strongly associated with trends like ‘MagSafe’, ‘Apple Wallet’, and
# ‘lidar sensor' than it is to “Biden” or “perfume”.

# # Approach: 
# The idea behind semantic search is to embed all entries in the corpus (sentences, paragraphs, or documents) into a vector space. At search time, the query is embedded into the same vector space and the closest embeddings from the corpus are found. These entries should have a high semantic overlap with the query.
# 
# I am breaking the challenge into below tasks:
# 1. Data Extraction
# 2. Data Cleaning
# 3. Set up Encoder
# 4. Create Index
# 5. Encode and Index
# 6. Search
# 
# Assumption: Search query - one word; Result - set of similar one-word keywords/trends

# # Task 1: Data Extraction
# To query our Google BigQuery data using Python, we need to connect the Python client to our BigQuery instance. We do so using a cloud client library for the Google BigQuery API. Here I am using google cloud bigquery library.

# install Google Cloud BigQuery library from command line if it is not already installed
# pip install --upgrade google-cloud-bigquery

from google.cloud import bigquery
from google.oauth2 import service_account

# get credentials from json file - provide path to json file if necessary
credentials = service_account.Credentials.from_service_account_file('nwo-sample-5f8915fdc5ec.json')

# connect the client to the database
project_id = 'nwo-sample'
client = bigquery.Client(credentials= credentials,project=project_id)


# Now that we have the BigQuery client set up and ready to use, we can execute queries on twitter and reddit datasets. For this, we use the query method, which inserts a query job into the BigQuery queue. These queries are then executed asynchronously. As soon as the job is complete, the method returns a Query_Job instance containing the results.
# sql query to get reddit posts dataset
reddit_query = """
    SELECT *
   FROM `nwo-sample.graph.reddit`
   LIMIT 100
"""
#save query results in a dataframe
reddit_df = client.query(reddit_query).to_dataframe()

# sql query to get twitter tweets dataset
tweets_query = """
    SELECT *
   FROM `nwo-sample.graph.tweets`
   LIMIT 100
"""
#save query results in a dataframe
tweets_df = client.query(tweets_query).to_dataframe()

# Examine first 5 rows of reddit posts data
reddit_df.head(5)

# Examine first 5 rows of twitter tweets data
tweets_df.head(5)


# Given a keyword, our task is to return the most relevant set of keywords. For this analysis, I am only considering tweets and body of reddit posts and storing them in separate dataframes
tweets = tweets_df.tweet
reddits = reddit_df.body


# # Task 2: Data Cleaning

#define method to clean tweets and body of reddit posts
import warnings
warnings.filterwarnings("ignore")

import html
import re
from string import punctuation

def clean(posts):
    
    for i in range (len(posts)):

        #remove newline “\n” 
        x = posts[i].replace("\n","")
        posts[i] = html.unescape(x)
        
        #remove all special characters and punctuation
        posts[i] = re.sub(r"[^A-Za-z0-9\s]+", "", posts[i])
    
        #convert to lower case
        posts[i] = posts[i].lower()
    
        #remove picture url
        posts[i] = re.sub(r'pictwittercom[\w]*',"", posts[i])
    
        #remove unicode represted spaces
        posts[i] = posts[i].replace('\xa0', '')
    
        #remove extra spaces, tabs and line breaks
        posts[i] = " ".join(posts[i].split())
    
        #remove non alphabetic characters
        posts[i] = " ".join([w for w in posts[i].split() if w.isalpha()])
    
        #remove short terms (single character)
        posts[i] = " ".join([w for w in posts[i].split() if len(w)>1])
        
    return posts

#clean tweets and reddit posts using the above method
cleaned_tweets = clean(tweets)
cleaned_reddits = clean(reddits)


# At this point, we already have much cleaner data, but there is one more thing that we need to do to make it even cleaner. In text-data contains insignificant words that are not used for the analysis process because they could mess up the analysis score. So, we’re about to clean them now using the nltk Python library. There are few steps that we need to follow to remove the stopwords: 
# 1. Preparing Stop words 
# 2. Tokenizing tweets and reddit posts 
# 3. Remove stop words from tokens

#To install nltk: pip3 install nltk

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Preparing stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# we exclude not from the stopwords corpus since removing not from the text will change the context of the text
stop_words.remove('not')

# define method to tokenize text and remove stop words
def tokenize_posts(posts_to_token):
    
    for i in range(len(posts_to_token)):
        posts_to_token[i] = word_tokenize(posts_to_token[i])
        
    for i in range(len(posts_to_token)):
        posts_to_token[i] = [word for word in posts_to_token[i] if not word in stop_words]

    return posts_to_token

#tokenize tweets and remove stopwords
tweets_to_token = cleaned_tweets.copy()
tokenized_tweets = tokenize_posts(tweets_to_token)

#tokenize reddit posts and remove stopwords
reddits_to_token = cleaned_reddits.copy()
tokenized_reddits = tokenize_posts(reddits_to_token)

#tweets before tokenzing
cleaned_tweets.head()

#tweets after tokenzing
tokenized_tweets.head()

#reddit posts before tokenzing
cleaned_reddits.head()

#reddit posts after tokenzing
tokenized_reddits.head()

#Store all tokenized tweets in a list and remove duplicates
import itertools
all_tweet_words = list(itertools.chain(*tokenized_tweets))
unique_tweet_words = list(set(all_tweet_words))

#Store all tokenized reddit posts in a list  and remove duplicates
all_reddit_words = list(itertools.chain(*tokenized_reddits))
unique_reddit_words = list(set(all_reddit_words))

#Store tweet tokens and reddit post tokens in a list  and remove duplicates
all_words = unique_tweet_words + unique_reddit_words
unique_words = list(set(all_words))


# # Task 3: Set up Encoder
# Now, let's set up an encoder to encode all the unique words into vectors. SentenceTransformers framework offers several pre-trained models that have been extensively evaluated for their quality to embedded search queries & paragraphs (Performance Semantic Search). Here I am using a pre-trained model (all-MiniLM-L6-v2) as it is 5 times faster than the best quality model and still offers good quality. 
# 
# More details at https://www.sbert.net/examples/applications/semantic-search/README.html

#To install SentenceTransformers: pip install -U sentence-transformers

#Set up encoder
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


# # Task 4: Create Indexer
# Now we will create FAISS indexer class which will store all embeddings efficiently for fast vector search.
# 
# FAISS (Facebook AI Similarity Search) is a library that allows developers to quickly search for embeddings of multimedia documents that are similar to each other.
# 
# More details at:https://ai.facebook.com/tools/faiss/

#To install faiss: conda install -c pytorch faiss-cpu

import faiss

class FAISS:
    def __init__(self, dimensions:int):
        self.dimensions = dimensions
        self.index = faiss.IndexFlatL2(dimensions)
        self.vectors = {}
        self.counter = 0
    
    def add(self, text:str, v:list):
        self.index.add(v)
        self.vectors[self.counter] = (text, v)
        self.counter += 1
        
    def search(self, v:list, k:int=10):
        distance, item_index = self.index.search(v, k)
        for dist, i in zip(distance[0], item_index[0]):
            if i==-1:
                break
            else:
                print(f'{self.vectors[i][0]}, %.2f'%dist)


# # Task 5: Encode and Index

#Install tqdm: pip3 install tqdm

#Create embeddings for all words and store them in FAISS. 
from tqdm import tqdm

dim = model.encode(['hello']).shape[-1]

index = FAISS(dim)
for q in tqdm(unique_words):
    emb = model.encode([q])
    index.add(q, emb)


# # Task 6: Search
# Similarity is measured by calculating the euclidean ditance between the search query and embeddings in the corpus. 
# Smaller distance implies stronger the association and vice versa
# 
# More details at https://faiss.ai/

#Define a search method which shows us the top k similar results (by default 10) given a query.
def search(s, k=10):
    emb = model.encode([s])
    print('keyword  distance')
    index.search(emb, k)

search('coronavirus')

