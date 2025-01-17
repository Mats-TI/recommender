from kmodes.kprototypes import KPrototypes
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

moviesData=pd.read_csv('./archive/tmdb_5000_movies.csv')
creditsData=pd.read_csv('./archive/tmdb_5000_credits.csv')

movieTitles=moviesData['original_title']
genres=moviesData['genres']
budgets=moviesData['budget']
runtimes=moviesData['runtime']

'''Since the data types are mixed, an improvement of the K-means algorithm called K-prototype, will be used'''

#print(moviesData.info()). A line that shows the data types of entries by column.

'''Preprocessing step: Check if the data has no empty values. In this case "homepage" column has over 3000 missing values
and may not be useful. To be removed'''
moviesData.drop(['homepage'],axis=1,inplace=True)
moviesData.dropna(inplace=True)
#print(moviesData.isna().sum()) #->Checks for null values

CategoricalColumnPositions=[moviesData.columns.get_loc(col) for col in list(moviesData.select_dtypes('object').columns)]
#Normalization using min-max (any other type could be used)
arrayCCP=np.array(CategoricalColumnPositions).reshape(-1,1)
normalizer=preprocessing.MinMaxScaler()
normalizedCCP=normalizer.fit_transform(arrayCCP).reshape(1,-1)

#Data conversion to matrix
movieMatrix=moviesData.to_numpy()

#The Elbow method to determine the optimal number of clusters (others may be used)
'''cost=[]
for i in range(1,10):
    kprototype=KPrototypes(n_jobs=-1,n_clusters=i,init='Huang',random_state=0)
    kprototype.fit_predict(movieMatrix,categorical=CategoricalColumnPositions)
    cost.append(kprototype.cost_)

costMatrix=pd.DataFrame({'Cluster':range(1,10),'cost':cost})
print(costMatrix)
costMatrix.plot(x="Cluster",y='cost')
plt.show()'''
''' ##The chosen number of clusters is 3, as per elbow method'''

kprototype=KPrototypes(n_jobs=-1,n_clusters=3,init='Huang',random_state=0)
kprototype.fit_predict(movieMatrix,categorical=CategoricalColumnPositions)

# These are the centroids of the three clusters
#print(kprototype.cluster_centroids_)

#Addition of the cluster to the dataFrame
moviesData['cluster_lables']=kprototype.labels_
moviesData['segment']=moviesData['cluster_lables'].map({0:'First',1:'Second',2:'Third'})

#Ordering of the clusters
moviesData['segment']=moviesData['segment'].astype('category')
moviesData['segment'].cat.reorder_categories(['First','Second','Third'])

## One-hot encoding is used to present categorical data to the ML model. Useful to select which columns are necessary, in this case all categorical columns selected.
ohe=preprocessing.OneHotEncoder()
encodedCategoricalData=ohe.fit_transform(moviesData[['status','title','genres','keywords','original_language','original_title','overview','production_companies','production_countries','release_date','spoken_languages','tagline','segment']]).toarray()

#Create labels for the encoded data
CategoricalDataLabels=np.array(ohe.categories_).ravel()
print(CategoricalDataLabels)

encodedDF=pd.DataFrame(encodedCategoricalData,columns=CategoricalDataLabels)
