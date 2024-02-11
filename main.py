import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


# configuration of plots
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# example text for model training (SMS messages)
# vocab creation
simple_train = ['call you tonight', 'Call me a cab', 'Please call me... PLEASE!']

# import countvectorizer class from sklearn to create and  vectorize vocab
vect = CountVectorizer()
vect.fit(simple_train)
print(vect.get_feature_names_out())

# transform training data into a 'document-term matrix'
simple_train_dtm = vect.transform(simple_train)
print(simple_train_dtm)

train_df = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names_out())

print("TRAIN DATASET")
print(train_df)


# example text for model testing
simple_test = ["please don't call me"]

# transform testing data into a document-term matrix (using existing vocabulary)
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()

test_df = pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names_out())

print('TEST DATAFRAME:')
print(test_df)

