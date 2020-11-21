# Using LDA to Cluster our Topics:

import numpy as np
import lda
import lda.datasets
import pandas as pd
X = pd.read_csv('/Users/luca/Offline/data/clean.csv')#lda.datasets.load_reuters()

vocs_list = []
X['Combined_Content'].apply(lambda x: vocs_list.append(x))
vocab = set(vocs_list)
titles = X['Combined_Content'].astype('str').dropna()
X.shape
X.sum()
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))