# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift, AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import multiprocessing

def attackIntance(label):

    # Define attack categories by dataset labels
    normal = ['normal']
    dos = ['neptune', 'back', 'teardrop', 'smurf', 'pod', 'land']
    probe = ['portsweep', 'satan', 'ipsweep', 'nmap']
    r2l = ['warezclient', 'multihop', 'ftp_write', 'imap', 'guess_passwd', 'warezmaster', 'spy', 'phf']
    u2r = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl']

    # Return the relevant attack category for each given label
    if label in normal:
        return 'normal'
    elif label in dos:
        return 'dos'
    elif label in probe:
        return 'probe'
    elif label in r2l:
        return 'r2l'
    elif label in u2r:
        return 'u2r'

def attackDF(df):

    # Apply attackIntance function to given dataframe
    df['attack'] = df.apply(lambda row: attackIntance(row.label), axis=1)

    # Return dataframe after attack categorisation
    return df

def parallelize_dataframe(df, func):

    # Number of cores on your machine
    num_cores = multiprocessing.cpu_count() - 1

    # Number of partitions to split dataframe
    num_partitions = num_cores

    # Split dataframe
    df_split = np.array_split(df, num_partitions)

    # Create multiprocesing pool using number of cores
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))

    # Close and join pool
    pool.close()
    pool.join()

    # Return merged dataframe
    return df

# Read dataframe
df = pd.read_csv('kddcup99_csv.csv')

# Assign attack categories by multiprocessing
df = parallelize_dataframe(df, attackDF)

df = df.drop_duplicates()

# Print dataframe shape
print('Dataframe Shape:', df.shape)

# Print dataframe columns
print('Columns:', list(df))

# Print unique counts of each label and attack category
print('\nUnique counts for each Label:\n', df.label.value_counts())
print('\nUnique counts for each Attack Category:\n', df.attack.value_counts())

df = df.loc[ (df['attack'] == 'normal') | (df['attack'] == 'dos')  ]
df = df.loc[ (df['label'] == 'normal') | (df['label'] == 'smurf') | (df['label'] == 'teardrop')| (df['label'] == 'back') ]
relevant = ['diff_srv_rate', 'dst_bytes', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'flag', 'rerror_rate', 'same_srv_rate', 'service', 'src_bytes', 'wrong_fragment', 'label', 'attack' ]
relevant2 = ['duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes', 'flag', 'land', 'wrong_fragment', 'urgent', 'count', 'serror_rate', 'rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_count', 'srv_serror_rate', 'label', 'attack']
df = df[relevant2]

# Choose n elements from each label
Samplesize = 206  #number of samples that you want
df = df.groupby('label', as_index=False).apply(lambda array: array.loc[np.random.choice(array.index, Samplesize, False),:])

# Store labels and delete them from dataset
labels = df['label']
del df['label']

# Store attack categories and delete them from dataset
attacks = df['attack']
del df['attack']

# Use one hot encoding for discrete/categorical variables
df = pd.get_dummies(df)

# dfNormal = df.loc[ (df['label'] == 'normal') ]
# dfNormalX = dfNormal.values
# model = KMeans(n_clusters=2, max_iter=1000)
# model.fit(dfNormalX)
# y = model.predict(dfNormalX)
# most_frequent_label = max(set((y).tolist()), key=y.tolist().count)
# print('Test:', most_frequent_label)

# print('##################################################################################################\n', df.head())

# Typecast all variables to float64
df = df.astype(np.float64)

# Get dataframe elements as nparray
X = df.values

# Standard Scalar
scaler = StandardScaler()
scaler.fit(X)

# # Apply PCA for 3 features
# pcaModel = PCA(n_components=3)
# pcaModel.fit(X)
# X = pcaModel.transform(X)

# Apply clustering by K-Means or Agglomerative clustering

model = KMeans(n_clusters=4, max_iter=1000, n_jobs=multiprocessing.cpu_count()-1)
model.fit(X)
y = model.predict(X)

# model = AgglomerativeClustering(n_clusters=4, linkage='single')
# model.fit(X)
# y = model.labels_

predicted_labels = (y).tolist()

# actual_labels = attacks.tolist()

actual_labels = labels.tolist()

# Create list of dictionaries to store both predicted and actual labels
dict_list = []

for i in range(len(predicted_labels)):
    row_dict = {}
    row_dict['Predicted'] = predicted_labels[i]
    row_dict['Actual'] = actual_labels[i]
    dict_list.append(row_dict)

# Create dataframe from list of dictionaries
new_dict = pd.DataFrame(dict_list)

# Print count of unique labels per each unique cluster
for predicted_label in new_dict['Predicted'].unique():
    print('----------------------------------------------------------------------\nPredicted Cluster:', predicted_label)
    filtered = new_dict.loc[new_dict['Predicted'] == predicted_label]
    print('\nUnique Counts:\n', filtered.Actual.value_counts())