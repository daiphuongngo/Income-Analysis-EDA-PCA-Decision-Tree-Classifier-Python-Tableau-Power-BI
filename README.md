# Income-Analysis-EDA-PCA-Decision-Tree-Classifier-Python-Tableau-Power-BI

## Overview:

Analyzing the background and income of surveyed adults

## Language and tools:

- Python

- Tableau

- Power BI

### Load libraries

```
import pandas as pd
import numpy as np 
import seaborn as sns
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
```

### Data Exploration 

### Data Preparation for Training

###  Split data into Train Set và Test Set (test_size=0.3)
```
from sklearn.model_selection import train_test_split
X_test, X_train, y_test, y_train = train_test_split(X_np, y_np, test_size=0.3, shuffle=True, random_state=1612)
```

### Use StandardScaler to scale X_train và X_test
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Apply PCA

```
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# noisy = np.random.normal(mnist.data, 4) # X = mnist.data
# pca = PCA(0.99).fit(noisy)

pca = PCA(0.99)
```

```
X_train_pca = pca.fit_transform(X_train_scaled)
print('Shape of X_train_pca:', X_train_pca.shape)

X_test_pca = pca.transform(X_test_scaled)
print('Shape of X_test_pca:', X_test_pca.shape)

num_comp = pca.n_components_
print('Number of components:', num_comp)
```

```
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
```

PCA Explained Variance Ratio - Cumulative Explained Variance vs Number of Components

<img src="https://user-images.githubusercontent.com/70437668/140874541-6570c441-4a76-4a50-ab8e-70b992d26728.jpg" width=50% height=50%>

### Decision Tree

```
import itertools
def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    print('')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.sca(ax)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
```

```
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
tree = DecisionTreeClassifier()
tree.fit(X_train_pca, y_train)
y_pred_tree = tree.predict(X_test_pca)

cm_tree = confusion_matrix(y_test, y_pred_tree)
fig, ax = plt.subplots()
plot_confusion_matrix(cm_tree, classes=np.unique(y), ax=ax,
                      title='Decision tree')
```

Confusion Matrix of Decision Tree

<img src="https://user-images.githubusercontent.com/70437668/140874598-df4f59a3-a942-4dbc-8485-2d3fe7896453.jpg" width=50% height=50%>

### Data Visualization in Python

#### Count plot for Occupation

![Count plot for Occupation](https://user-images.githubusercontent.com/70437668/140842997-7ddda2d0-cf89-4aae-a38a-d85d52a4acba.jpg)

#### Count plot for Income vs Occupation

![Count plot for Income vs Occupation](https://user-images.githubusercontent.com/70437668/140843032-03910397-e724-47ba-9d89-636daf35c009.jpg)

#### Count plot for Income by Labels

<img src="![Count plot for Income by Labels](https://user-images.githubusercontent.com/70437668/140843059-ea66e192-4d4d-4002-a496-58e32c696952.jpg)" width=50% height=50%>

#### Dashboard - Income by Occupation of selected Countries (Radar chart)

![Dashboard - Income by Occupation of selected Countries](https://user-images.githubusercontent.com/70437668/141012402-63cfa6c3-ca8e-493f-9a6f-8755c336be60.jpg)

#### Dashboard - Income, Race

![Dashboard - Income, Race](https://user-images.githubusercontent.com/70437668/140843027-abf72a9b-a91d-4125-9af0-908cafebf2e9.jpg)

#### Dashboard - Income by Occupation, Relationship, Marital Status, Workclass, Education

![Dashboard - Income by Occupation, Relationship, Marital Status, Workclass, Education](https://user-images.githubusercontent.com/70437668/140873891-9590d3b9-b014-404a-a3ed-ef4061afea3b.jpg)

#### Dashboard - Income by Occupation, Gender, Countries, Race

![Dashboard - Income by Occupation, Gender, Countries, Race](https://user-images.githubusercontent.com/70437668/140873906-0a9df1f4-983d-4ca6-aa33-f378b7197cad.jpg)

#### Dashboard - Income by Occupation, Gender, Age

![Dashboard - Income by Occupation, Gender, Age](https://user-images.githubusercontent.com/70437668/140873923-caca6e77-4748-43d9-937d-35b0500a4dca.jpg)

#### Dashboard - Average Hours per Week

![Dashboard - Average Hours per Week](https://user-images.githubusercontent.com/70437668/140873940-e3c90577-7954-4657-9ffe-b64d0f199c4b.jpg)

