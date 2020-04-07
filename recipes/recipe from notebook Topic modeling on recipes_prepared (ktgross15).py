# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Topic Modeling on recipes_prepared

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Topic models are statistical models that aim to discover the 'hidden' thematic structure in a collection of documents, i.e. identify possible topics in our corpus. It is an interative process by nature, as it is crucial to determine the right number of topics.
# 
# This notebook is organised as follows:
# 
# * [Setup and dataset loading](#setup)
# * [Text Processing:](#text_process) Before feeding the data to a machine learning model, we need to convert it into numerical features.
# * [Topics Extraction Models:](#mod) We present two differents models from the sklearn library: NMF and LDA.
# * [Topics Visualisation with pyLDAvis](#viz)
# * [Topics Clustering:](#clust)  We try to understand how topics relate to each other.
# * [Further steps](#next)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Setup and dataset loading <a id="setup" />
# 
# First of all, let's load the libraries that we'll use.
# 
# **This notebook requires the installation of the [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/readme.html#installation) package.**
# [See here for help with intalling python packages.](https://www.dataiku.com/learn/guide/code/python/install-python-packages.html)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# %pylab inline
import warnings                         # Disable some warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd,  seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text

from sklearn.decomposition import LatentDirichletAllocation,NMF
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dataset_limit = 100000

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# The first thing we do is now to load the dataset and identify possible text columns.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Take a handle on the dataset
mydataset = dataiku.Dataset("recipes_prepared")

# Load the first lines.
# You can also load random samples, limit yourself to some columns, or only load
# data matching some filters.
#
# Please refer to the Dataiku Python API documentation for more information
df = mydataset.get_dataframe(limit = dataset_limit)

df_orig = df.copy()

# Get the column names
numerical_columns = list(df.select_dtypes(include=[np.number]).columns)
categorical_columns = list(df.select_dtypes(include=[object]).columns)
date_columns = list(df.select_dtypes(include=['<M8[ns]']).columns)

# Print a quick summary of what we just loaded
print "Loaded dataset"
print "   Rows: %s" % df.shape[0]
print "   Columns: %s (%s num, %s cat, %s date)" % (df.shape[1],
                                                    len(numerical_columns), len(categorical_columns),
                                                    len(date_columns))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# By default, we suppose that the text of interest for which we want to extract topics is the first of the categorical columns.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
raw_text_col = 'ingredients'

raw_text = df[raw_text_col]
# Issue a warning if data contains NaNs
if(raw_text.isnull().any()):
    print('\x1b[33mWARNING: Your text contains NaNs\x1b[0m')
    print('Please take care of them, the countVextorizer will not be able to fit your data if it contains empty values.')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Text Processing <a id="text_process" />
# 
# We cannot directly feed the text to the Topics Extraction Algorithms. We first need to process the text in order to get numerical vectors. We achieve this by applying either a CountVectorizer() or a TfidfVectorizer(). For more information on those technics, please refer to thid [sklearn documentation](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html).

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# As with any text mining task, we first need to remove stop words that provide no useful information about topics. *sklearn* provides a default stop words list for english, but we can alway add to it any custom stop words : <a id="stop_words" /a>

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
custom_stop_words = ['tablespoon','teaspoon','cup','ounc','pound','inch']
#                      ,'chop','cut','slice','dice','minc']

stop_words = text.ENGLISH_STOP_WORDS.union(custom_stop_words)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
tfidf_vectorizer = TfidfVectorizer(strip_accents = 'unicode',stop_words = stop_words,lowercase = True,
                                token_pattern = r'\b[a-zA-Z]{3,}\b', max_df = 0.75, min_df = 0.02)

text_tfidf = tfidf_vectorizer.fit_transform(raw_text)

print(text_tfidf.shape)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Topics Extraction Models <a id="mod" />
# 
# There are two very popular models for topic modelling, both available in the sklearn library:
# 
# * [NMF (Non-negative Matrix Factorization)](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization),
# * [LDA (Latent Dirichlet Allocation)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
# 
# Those two topic modeling algorithms infer topics from a collection of texts by viewing each document as a mixture of various topics. The only parameter we need to choose is the number of desired topics `n_topics`.
# It is recommended to try different values for `n_topics` in order to find the most insightful topics. For that, we will show below different analyses (most frequent words per topics and heatmaps).

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
n_topics = 10

topics_model = LatentDirichletAllocation(n_topics, random_state=0)
# topics_model = NMF(n_topics, random_state=0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
topics_model.fit(text_tfidf)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Can tune number oftopics

topics_model.score(text_tfidf)

# topics_model.get_params()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create Document — Topic Matrix
lda_output = topics_model.transform(text_tfidf)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# column names
# CHANGE LATER!
topicnames = ['Topic' + str(i) for i in range(topics_model.n_components)]
# topicnames = ['Mexican', 'smoothies', ]

# index names
docnames = ['Doc' + str(i) for i in range(len(df))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

df_document_topic.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df['topic'] = list(df_document_topic['dominant_topic'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.tail()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Most Frequent Words per Topics
# An important way to assess the validity of our topic modelling is to directly look at the most frequent words in each topics.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
n_top_words = 12
feature_names = tfidf_vectorizer.get_feature_names()

def get_top_words_topic(topic_idx):
    topic = topics_model.components_[topic_idx]

    print( [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] )

for topic_idx, topic in enumerate(topics_model.components_):
    print ("Topic #%d:" % topic_idx )
    get_top_words_topic(topic_idx)
    print ("")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Pay attention to the words present, if some are very common you may want to go back to the [definition of custom stop words](#stop_words).

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Naming the topics
# 
# Thanks to the above analysis, we can try to name each topics:

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dict_topic_name = {i: "topic_"+str(i) for i in xrange(n_topics)}
dict_topic_name = {0: 'Mexican'}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Topics Heatmaps
# 
# Another visual helper to better understand the found topics is to look at the heatmap for the document-topic and topic-words matrices. This gives us the distribution of topics over the collection of documents and the distribution of words over the topics.
# We start with the topic-word heatmap where the darker the color is the more the word is representative of the topic:

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
word_model = pd.DataFrame(topics_model.components_.T)
word_model.index = feature_names
word_model.columns.name = 'topic'
word_model['norm'] = (word_model).apply(lambda x: x.abs().max(),axis=1)
word_model = word_model.sort_values(by='norm',ascending=0) # sort the matrix by the norm of row vector
word_model.rename(columns = dict_topic_name, inplace = True) #naming topic

del word_model['norm']

# plt.figure(figsize=(9,8))
# sns.heatmap(word_model[:10])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# We now display the document-topic heatmap:

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# retrieve the document-topic matrix
document_model = pd.DataFrame(topics_model.transform(text_tfidf))
document_model.columns.name = 'topic'
document_model.rename(columns = dict_topic_name, inplace = True) #naming topics

# plt.figure(figsize=(9,8))
# sns.heatmap(document_model.sort_index()[:10]) #we limit here to the first 10 texts

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Topic distribution over the corpus
# We can look at how the topics are represented in the collection of documents.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
topics_proportion = document_model.sum()/document_model.sum().sum()
topics_proportion.plot(kind = "bar")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# For each topic, we can investigate the documents the most representative for the given topic:

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def top_documents_topics(topic_name, n_doc = 3, excerpt = True):
    '''This returns the n_doc documents most representative of topic_name'''

    document_index = list(document_model[topic_name].sort_values(ascending = False).index)[:n_doc]
    for order, i in enumerate(document_index):
        print "Text for the {}-th most representative document for topic {}:\n".format(order + 1,topic_name)
        if excerpt:
            print raw_text[i][:1000]
        else:
            print raw_text[i]
        print "\n******\n"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# top_documents_topics("Topic #1")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Topics Visualization with pyLDAvis <a id="viz">
# 
# Thanks to the pyLDAvis package, we can easily visualise and interpret the topics that has been fit to our corpus of text data.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# pyLDAvis.sklearn.prepare(topics_model, text_tfidf, tfidf_vectorizer)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Topics Clustering  <a id="clust">
# 
# Once we have fitted topics on the text data, we can try to understand how they relate to one another: we achieve this by doing a hierachical clustering on the topics. We propose two methods, the first is based on a correlation table between topics, the second on a contigency table.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# correlation matrix between topics
cor_matrix = np.corrcoef(document_model.iloc[:,:n_topics].values,rowvar=0)

#Renaming of the index and columns
cor_matrix = pd.DataFrame(cor_matrix)
cor_matrix.rename(index = dict_topic_name, inplace = True)
cor_matrix.rename(columns= dict_topic_name, inplace = True)

sns.clustermap(cor_matrix, cmap="bone")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# contingency table on the binarized document-topic matrix
document_bin_topic = (document_model.iloc[:,:n_topics] > 0.25).astype(int)
contingency_matrix = np.dot(document_bin_topic.T.values, document_bin_topic.values )

#Renaming of the index and columns
# contingency_matrix = pd.DataFrame(contingency_matrix)
# contingency_matrix.rename(index = dict_topic_name, inplace = True)
# contingency_matrix.rename(columns= dict_topic_name, inplace = True)

# sns.clustermap(contingency_matrix)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Further steps  <a id="next">
# 
# Topics extraction is a vast subject and a notebook can only show so much. There still much thing we could do, here are some ideas:

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### 1. Discard documents from noise topics
# The following helper function takes as argument the topics for which we wish to discard documents.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def remove_doc(*topic_name):

    doc_max_topic = document_model.idxmax(axis = 1)
    print "Removing documents whose main topic are in ", topic_name
    doc_max_topic_filtered = doc_max_topic[~doc_max_topic.isin(topic_name)]
    return [raw_text[i] for i in doc_max_topic_filtered.index.tolist()]

#E.g.: to remove documents whose main topic are topic_1 or topic_3, we would simply call remove_doc("topic_0","topic_2")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# For the 20newsgroup dataset, try this to remove text of topic "Misc"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#raw_text_filtered = remove_doc("Misc")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### 2. Scoring the topic model on new text
# Finally, we can score new text with our topic model as follows.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# raw_text

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
new_text = raw_text[:3] #Change this to the new text you'd like to score !

tfidf_new_text = tfidf_vectorizer.transform(new_text)
result = pd.DataFrame(topics_model.transform(tfidf_new_text), columns = [dict_topic_name[i] for i in xrange(n_topics)])
sns.heatmap(result)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
recipes_topic_modeling = dataiku.Dataset("recipes_topic_modeling")
recipes_topic_modeling.write_with_schema(df)