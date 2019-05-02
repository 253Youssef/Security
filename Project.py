#!/usr/bin/env python
# coding: utf-8

# # Libraries & Modules

# In[1]:


import pandas as pd
import numpy as np
np.random.seed(None)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import multiprocessing


# # Functions

# ## Assign Attack Category

# In[2]:


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


# ## Parallelise Dataframe Operations

# In[3]:


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


# ## Cluster & Compare Labels

# In[4]:


def ClusterANDCompareOptimised(df, attack, num_clusters):

	# Get instances with the label normal
	df_filtered = df.loc[(df['attack'] == attack)]

	labels = df_filtered['label']
	del df_filtered['label']

	attacks = df_filtered['attack']
	del df_filtered['attack']

	df_filtered_values = df_filtered.values

	# Standard Scalar
	scaler = StandardScaler()
	df_filtered_values_scaled = scaler.fit_transform(df_filtered_values)

	# Apply PCA for 3 features
	pca_model = PCA(n_components=3)
	pca_model.fit(df_filtered_values_scaled)
	df_filtered_values_pca = pca_model.transform(df_filtered_values_scaled)

	model = KMeans(n_clusters=num_clusters, max_iter=1000)
	model.fit(df_filtered_values_pca)
	y = model.predict(df_filtered_values_pca)

	predicted_labels = (y).tolist()

	actual_labels = labels.tolist()

	df_filtered['predicted'] = predicted_labels
	df_filtered['label'] = labels
	df_filtered['attack'] = attacks

	# Print count of unique labels per each unique cluster
	for predicted_label in df_filtered['predicted'].unique():
		print('----------------------------------------------------------------------\nPredicted Cluster:', predicted_label)
		filtered = df_filtered.loc[df_filtered['predicted'] == predicted_label]
		print('\nUnique Counts:\n', filtered.label.value_counts())

	return actual_labels, predicted_labels, scaler, pca_model, df_filtered


# ## Testing

# In[9]:


def testing(df, label, model,id):

	df_filtered = df.loc[df['label']==label]
	del df_filtered['label']
	del df_filtered['attack']

	df_filtered_values = df_filtered.values

	# Standard Scalar
	scaler = StandardScaler()
	df_filtered_scaled = scaler.fit_transform(df_filtered_values)

	# Apply PCA for 3 features
	pca_model = PCA(n_components=3)
	pca_model.fit(df_filtered_scaled)
	df_normal_test_pca = pca_model.transform(df_filtered_scaled)

	predicted_labels = model.predict(df_normal_test_pca)

	values, counts = np.unique(predicted_labels, return_counts=True)
	# print(counts)
	global Samplesize
	c = [0,0,0,0]
	# print(values,counts)
	for value in values:
		c[value] = counts[np.where(values == value)[0][0]]
	# print(c)
	precision = c[id]/sum(c)
	recall = c[id]/df.label.tolist().count(label)
	# print(df.label.tolist().count(label))
	# print("\nprecision:",precision,"  recall:",recall)

	return list(zip(values, counts)),precision,recall


# # Code

# ## Read Dataframe and get Relevant Information

# In[10]:


# Read dataframe
df = pd.read_csv('kddcup99_csv.csv')

# Assign attack categories by multiprocessing
df = parallelize_dataframe(df, attackDF)

# Drop duplicate data instances
df = df.drop_duplicates()

# Get datapoints with attack categories of normal and dos
df = df.loc[(df['attack'] == 'normal') | (df['attack'] == 'dos')]

labels = df['label']
attacks = df['attack']
del df['label']
del df['attack']

df = pd.get_dummies(df)

df['label'] = labels
df['attack'] = attacks

# Print dataframe shape
print('Dataframe Shape:', df.shape)

# Print unique counts of each label and attack category
print('\nUnique counts for each Label:\n', df.label.value_counts())
print('\nUnique counts for each Attack Category:\n', df.attack.value_counts())

# See dataframe head
df.head()


# ## Normal Instances Analysis

# In[11]:


normal_actual_labels, normal_predicted_labels, normal_scaler, normal_pca_model, normal_df_clustered = ClusterANDCompareOptimised(df, 'normal', 10)


# In[12]:


# Filter out anaomly normal
normal_df_clustered = normal_df_clustered.loc[normal_df_clustered['predicted']==0]
del normal_df_clustered['predicted']
normal_df_clustered.head()


# # DoS Attacks Analysis

# In[13]:


dos_actual_labels, dos_predicted_labels, dos_scaler, dos_pca_model, dos_df_clustered = ClusterANDCompareOptimised(df, 'dos', 6)


# In[14]:


df_dos_filtered = df.loc[(df['label'] == 'pod')|(df['label'] == 'back')|(df['label'] == 'teardrop')]
dos_actual_labels_filtered, dos_predicted_labels_filtered, dos_scaler_filtered, dos_pca_model_filtered, dos_df_clustered_filtered = ClusterANDCompareOptimised(df_dos_filtered, 'dos', 3)
del dos_df_clustered_filtered['predicted']


# In[15]:


# Print unique counts of each label
print('\nUnique counts for each Label:\n', dos_df_clustered_filtered.label.value_counts())

dos_df_clustered_filtered.head()


# ## Cluster Datapoints into Normal & DoS

# In[16]:


# Choose n elements from each label
Samplesize = 200  #number of samples that you want
normal_df_clustered = normal_df_clustered.groupby('label', as_index=False).apply(lambda array: array.loc[np.random.choice(array.index, Samplesize, False),:])

# Choose n elements from each label
dos_df_clustered_filtered = dos_df_clustered_filtered.groupby('label', as_index=False).apply(lambda array: array.loc[np.random.choice(array.index, Samplesize, False),:])

normal_labels = normal_df_clustered['label']
normal_attacks = normal_df_clustered['attack']
del normal_df_clustered['label']
del normal_df_clustered['attack']

dos_labels = dos_df_clustered_filtered['label']
dos_attacks = dos_df_clustered_filtered['attack']
del dos_df_clustered_filtered['label']
del dos_df_clustered_filtered['attack']

labels = normal_labels.tolist() + dos_labels.tolist()
attacks = normal_attacks.tolist() + dos_attacks.tolist()


# In[17]:


normal_df_clustered_values = normal_df_clustered.values
normal_df_clustered_values_scaled = normal_scaler.transform(normal_df_clustered_values)
normal_df_clustered_values_pca = normal_pca_model.transform(normal_df_clustered_values_scaled)


dos_df_clustered_filtered_values = dos_df_clustered_filtered.values
dos_df_clustered_filtered_values_normal_scaled = dos_scaler_filtered.transform(dos_df_clustered_filtered_values)
dos_df_clustered_filtered_values_normal_pca = dos_pca_model_filtered.transform(dos_df_clustered_filtered_values_normal_scaled)

concatenated_data = np.concatenate((normal_df_clustered_values_pca,dos_df_clustered_filtered_values_normal_pca))

model = KMeans(n_clusters=4, max_iter=2)
model.fit(concatenated_data)
y = model.predict(concatenated_data)

predicted_labels = (y).tolist()

actual_labels = labels

df_filtered = pd.DataFrame()

df_filtered['predicted'] = predicted_labels
df_filtered['label'] = labels
df_filtered['attack'] = attacks

def calcPrecisionRecall(label):
	global Samplesize
	if len(label)==1:
		return 1,label[0]/Samplesize
	else:
		temp = sum(label)
		return label[0]/temp,label[0]/Samplesize


totalprecision = 0
totalrecall = 0
labels = dict()
# Print count of unique labels per each unique cluster
for predicted_label in df_filtered['predicted'].unique():
	print('----------------------------------------------------------------------\nPredicted Cluster:', predicted_label)
	filtered = df_filtered.loc[df_filtered['predicted'] == predicted_label]
	print('\nUnique Counts:\n', filtered.label.value_counts())
	# print("precision -->",calcPrecisionRecall(filtered.label.value_counts()),"<-- recall\n")
	labels[(max(set(filtered.label.tolist()), key = filtered.label.tolist().count))] =predicted_label
	(p,r) = calcPrecisionRecall(filtered.label.value_counts())
	totalprecision+=p
	totalrecall+=r

# ## Testing Datapoints

# In[18]:

print("\n\n\n\nclustering precision:",totalprecision/4," clustering recall:",totalrecall/4)

z1,p1,r1 = testing(df, 'normal', model,labels['normal'])


# In[19]:


z2,p2,r2 = testing(df, 'back', model,labels['back'])


# In[20]:


z3,p3,r3 = testing(df, 'teardrop', model,labels['teardrop'])


# In[21]:


z4,p4,r4 = testing(df, 'pod', model,labels['pod'])

# print(p1,p2,p3,p4,"\n",r1,r2,r3,r4)

print("\n\n\n\nmodel precision:",(p1+p2+p3+p4)/4," model recall:",(r1+r2+r3+r4)/4)

# In[ ]:
