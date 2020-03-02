"""
(2) Loading the Data

Working with the iris flowers dataset is the "hello world" equivalent in ML and Statistics
"""

# 2.1 Import Libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 2.2 Load Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = [
    'sepal-length',
    'sepal-width',
    'petal-length',
    'petal-width',
    'class'
]
dataset = read_csv(url, names=names)

"""
(3) Summarize dataset

The data is going to be looked at in a few different ways:
* Dimensions of the dataset
* Peek at the data itself
* Statistical summary of all attributes
* Breakdown of the data by the class variable
"""

# 3.1 Dimensions of Dataset
# Shape
print(dataset.shape)

# 3.2 Peeking at the Data
print(dataset.head(20))

# 3.3 Statistical Summary
print(dataset.describe())

# 3.4 Class Distribution
print(dataset.groupby('class').size())

"""
(4) Data Visualization

Visualize the data that was presented in (3) in two types of plots:
* Univariate plots (better understand each attribute)
* Multivariate plots (better understand relationships between attributes)
"""

# 4.1 Univariate Plots
# Box and whisker plots
dataset.plot(kind='box',
             subplots=True,
             layout=(2, 2),
             sharex=False,
             sharey=False)
pyplot.show()

# Histogram
dataset.hist()
pyplot.show()

# 4.2 Multivariate Plots
# Scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

"""
(5) Evaluate Algorithms

Create models of data and estimate accuracy on unseen data. This covers:
* Separating out validation dataset
* Setup of test harness to use 10-fold cross validation
* Build of multiple different models to predict species from flower measurements
* Selecting the best model
"""

# 5.1 Creating Validation Dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

# 5.3 Build Models
# Spot-checking algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# 5.4 Selecting the Best Model
# Compare algorithms to check for the best model
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm comparison')
pyplot.show()

"""
(6) Making Predictions

Choosing an algorithm to use for making the predictions.
From previous results, SVM seems to be most accurate.
"""

# 6.1 Make Predictions
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# 6.2 Evaluate Predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
