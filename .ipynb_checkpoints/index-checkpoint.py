import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler

normal_dataset_path = r'Images/new_normal/'
cataract_dataset_path = r'Images/new_cataract_copy/'


file_normal = 1375
file_cataract = 937

width,height = 400,400
distance = 10
teta = 90
data_eye = np.zeros((7, 2310))
count = 0
indextable = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'ASM', 'correlation', 'Label']

def get_feature(matrix, name):
    feature = graycoprops(matrix, name)
    result = np.average(feature)
    return result

def preprocessingImage(image):
    test_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    test_img_gray = cv.cvtColor(test_img, cv.COLOR_RGB2GRAY)
    test_img_thresh = cv.adaptiveThreshold(test_img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,11,3)
    cnts = cv.findContours(test_img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        test_img_ROI = test_img[y:y+h, x:x+w]
        break
    test_img_ROI_resize = cv.resize(test_img_ROI, (width, height))
    test_img_ROI_resize_gray = cv.cvtColor(test_img_ROI_resize, cv.COLOR_RGB2GRAY)
    
    return test_img_ROI_resize_gray


def populate(path, size, label):
    global count
    for file in range(1, size):
        image = cv.imread(f'{path}{str(file).zfill(4)}.jpg')
        img = preprocessingImage(image)
    
        glcm = graycomatrix(img, [distance], [teta], levels=256, symmetric=True, normed=True)
    
        for i in range(len(indextable[:-1])):
            features = []
            feature = get_feature(glcm, indextable[i])
            features.append(feature)
            data_eye[i, count] = features[0]
        data_eye[len(indextable) - 1, count] = label
    
        count = count + 1



populate(normal_dataset_path,file_normal,0)

populate(cataract_dataset_path,file_cataract,1)

data = pd.DataFrame(np.transpose(data_eye), columns = indextable)
scaler = MinMaxScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
data.to_csv("cataract_data1.csv")
print(data.head())

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20,30],
            'kernel': ['rbf','linear','poly']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10,50,100]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10,50,100]
        }
    },
    'KNN' : {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3,7,11,13]
        }
    }
}


from sklearn.model_selection import GridSearchCV
def test_model(X, y):
    scores = []

    for model_name, mp in model_params.items():
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        clf.fit(X, y)
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })   
    scores = sorted(scores, key = lambda x : x.get('best_score'), reverse=True)
    
    df_score = pd.DataFrame(scores,columns=['model','best_score','best_params'])
    print(df_score)

X = data.drop(['Label'], axis='columns')
y = data.Label
# test_model(X, y)    