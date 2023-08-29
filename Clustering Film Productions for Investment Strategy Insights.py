import pandas as pd

# command below ensures matplotlib output can be included in Notebook
%matplotlib inline

df = pd.read_csv('top-500-movies.csv') # from Koggle
# df.head() # view data
# df.info() # check if null, data type

df['mpaa'].unique()
df['genre'].unique()

df = df[df.mpaa != 'Unrated']
df = df.dropna(subset=['mpaa','genre'])

# Preserve an original version of dataset for future reference use
OG_dataset = df.iloc[:,:].values
'''
Better mapping method?
'''

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
df['mpaa'].value_counts().plot.bar(figsize=(5, 4),title='MPAA')
plt.show()
df['genre'].value_counts().plot.bar(figsize=(5, 4),title='Genre')
plt.show()

del df['rank']
del df['release_date']
del df['title']
del df['url']
del df['domestic_gross']  # could be selected as a feature for pure classification for movies
del df['worldwide_gross'] # could be selected as a feature for pure classification for movies
del df['opening_weekend'] # could be selected as a feature for gross prediction after the opening week

# df.info() # check current null

X = df.iloc[:,:].values

# Fill in numeric null data
import numpy as np
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="median")
imputer = imputer.fit(X[:,[0,3,4,5]])
X[:,[0,3,4,5]] = imputer.transform(X[:,[0,3,4,5]])

# Be careful with the not-null but 0 values
imputer = SimpleImputer(missing_values=0,strategy="median")
imputer = imputer.fit(X[:,[0,3,4,5]])
X[:,[0,3,4,5]] = imputer.transform(X[:,[0,3,4,5]])


# One-hot encoder for object datatype
# Do it backwards for error-free concatenation
# genre
ary_dummies = pd.get_dummies(X[:,2]).values
ary_dummies = np.concatenate((X[:,:2],ary_dummies),axis=1)
X = np.concatenate((ary_dummies,X[:,3:]),axis=1)
# mpaa
ary_dummies = pd.get_dummies(X[:,1]).values
ary_dummies = np.concatenate((X[:,:1],ary_dummies),axis=1)
X = np.concatenate((ary_dummies,X[:,2:]),axis=1)

X_tree = X # preserve an original version X for Dicision Tree


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler().fit(X)
X = sc_X.transform(X)

from sklearn.decomposition import PCA

# Test the number of PCA components
pca = PCA(n_components=None)
pca.fit(X)
info_covered = pca.explained_variance_ratio_
cumulated_sum = np.cumsum(info_covered)
plt.plot(cumulated_sum, color="blue")
plt.title("PCA")
plt.xlabel("Components")
plt.ylabel("Ratio")
plt.show()

# Set components to 2 for k-means to find best K
pca = PCA(n_components=2)
X = pd.DataFrame(pca.fit_transform(X))

from sklearn.cluster import KMeans

# Find Best K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Draw WCSS for each K
import matplotlib.pyplot as plt

plt.plot(range(1, 11), wcss)
plt.scatter(3, wcss[2], s = 200, c = 'red', marker='*')
plt.title("The Best K")
plt.xlabel("# of Clusters")
plt.ylabel("WCSS")
plt.show()

# Draw clusters with the value of 3
kmeans = KMeans(n_clusters=3, init="k-means++", random_state=0)
Y = kmeans.fit_predict(X)

# Draw Samples
Y_array = Y.ravel()
plt.scatter(X.iloc[Y_array==0, 0], X.iloc[Y_array==0, 1], s=50, c="red", label="Cluster 0")
plt.scatter(X.iloc[Y_array==1, 0], X.iloc[Y_array==1, 1], s=50, c="blue", label="Cluster 1")
plt.scatter(X.iloc[Y_array==2, 0], X.iloc[Y_array==2, 1], s=50, c="green", label="Cluster 2")

# Labels & Legends
plt.title("Clusters of Productions")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(loc="best")
plt.show()
'''
plot clusters count
'''



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,random_state=0)


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

print(classification_report(Y_test, Y_pred))
labels = unique_labels(Y_test)
col = [f'Predicted {label}' for label in labels]
ind = [f'Actual {label}' for label in labels]
table = pd.DataFrame(confusion_matrix(Y_test, Y_pred), columns=col, index=ind)
sns.heatmap(table, annot=True, fmt='d', cmap='Blues')
plt.show()

# K-Fold Validation
from sklearn.model_selection import cross_val_score

k_fold = 10
accuracies = cross_val_score(estimator=classifier, X=X, y=Y.ravel(),
                             scoring="accuracy", cv=k_fold, n_jobs=-1)
print("{} Folds Mean Accuracy: {:.2%}".format(k_fold, accuracies.mean()))


from sklearn import tree

model = classifier.fit(X_tree, Y_array)
fig = plt.figure(figsize=(25,20))
graph = tree.plot_tree(classifier, filled=True)
plt.show()

# clearer graph in text
text_representation = tree.export_text(classifier)
print(text_representation)