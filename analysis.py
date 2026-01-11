from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

path='cataract_data1.csv'

original = pd.read_csv(path)
original.drop(["Unnamed: 0"], axis=1, inplace=True)
data = original.copy()

count_learning_curves = 0
count_validation_curves = 0

def my_learning_curve(X, y, model, model_name):
    train_sizes, train_scores, test_scores = learning_curve( estimator=model, X = X, y = y, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([-0.3, 1.3])
    plt.title(model_name)

    plt.savefig(f"images\\{count_learning_curves}_Learning_Curve_{model_name}.png")
    plt.show()

def my_validation_curve(X, y, model, name, param_range, model_name):
    train_scores, test_scores = validation_curve(estimator=model, X=X, y=y, param_name=name, param_range=param_range, cv=10)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    
    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    
    plt.title(model_name)
    plt.grid()
    plt.legend(loc='lower right')
    plt.xlabel(name)
    plt.ylabel('Accuracy')
    plt.ylim([-0.3, 1.3])
    
    plt.savefig(f"images\\{count_validation_curves}_Validation_Curve_{model_name}.png")

    plt.show()    

def show_curve(X, y):
    global count_learning_curves
    
    svmModel = svm.SVC(gamma='auto')
    knn = KNeighborsClassifier(11)
    lr = LogisticRegression(solver='liblinear',multi_class='auto')
    rf = RandomForestClassifier()
    nb = GaussianNB()

    my_learning_curve(X, y, svmModel, "SVM")
    my_learning_curve(X, y, knn, "KNN")
    my_learning_curve(X, y, lr, "Logistic Regression")
    my_learning_curve(X, y, rf, "Random Forest")
    my_learning_curve(X, y, nb, "Naive Bayes")
    
    count_learning_curves += 1    

def show_validation(X, y):
    global count_validation_curves
    
    svmModel = svm.SVC(gamma='auto')
    param_range_svm = np.logspace(-6, -1, 5)
    
    knn = KNeighborsClassifier()
    param_range_knn = [1, 3, 5, 7, 9, 11]
    
    lr = LogisticRegression()
    param_range_lr = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    my_validation_curve(X, y, svmModel, "gamma", param_range_svm, "SVM")
    my_validation_curve(X, y, knn, "n_neighbors", param_range_knn, "KNN")
    my_validation_curve(X, y, lr, 'C', param_range_lr, "Logistic Regression")
    
    count_validation_curves += 1

X = data.drop(['Label'], axis='columns')
y = data.Label
show_validation(X, y)  
original.drop_duplicates(subset=None, keep='first', inplace=True)
data = original.copy()


cor_matrix = data.corr().abs()
print(cor_matrix)
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]

original = original.drop(to_drop, axis=1)
data = data.drop(to_drop, axis=1)
print(data.info())



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.15, random_state=0)

knn = make_pipeline(
    KNeighborsClassifier(11)
)

knn.fit(X_train, y_train)
joblib.dump(knn, 'knn1.pkl')
knn_model = joblib.load("knn1.pkl")
print(knn_model.score(X_test, y_test))

svm = make_pipeline(
    svm.SVC(gamma='auto',C=30,kernel='rbf')
)
svm.fit(X_train,y_train)
joblib.dump(svm, 'svm1.pkl')
svm_model = joblib.load("svm1.pkl")
print(svm_model.score(X_test, y_test))

lr = make_pipeline(

    LogisticRegression(solver='liblinear',multi_class='auto')
)

lr.fit(X_train,y_train)
joblib.dump(lr, 'lr1.pkl')
lr_model = joblib.load("lr1.pkl")
print(lr_model.score(X_test, y_test))

rfc = make_pipeline(
    RandomForestClassifier(n_estimators = 50, max_depth = 5, max_features= 'sqrt')
)

rfc.fit(X_train,y_train)
joblib.dump(rfc, 'rfc1.pkl')
rfc_model = joblib.load("rfc1.pkl")
print(rfc_model.score(X_test, y_test))

nb = make_pipeline(
    GaussianNB()
)

nb.fit(X_train,y_train)
joblib.dump(nb, 'nb1.pkl')
nb_model = joblib.load("nb1.pkl")
print(nb_model.score(X_test, y_test))

