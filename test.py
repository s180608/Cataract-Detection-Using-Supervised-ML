import numpy as np
import pandas as pd
import cv2 as cv
from skimage.feature import graycomatrix, graycoprops
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split

indextable = ['dissimilarity', 'contrast', 'homogeneity', 'energy','ASM', 'correlation', 'Label']
width, height = 400, 400
distance = 10
teta = 90
path='cataract_data1.csv'

original = pd.read_csv(path)
original.drop(["Unnamed: 0"], axis=1, inplace=True)
data = original.copy()
X = data.drop(['Label'], axis='columns')
y = data.Label
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.15, random_state=0)

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

def extract(image):
    data_eye = np.zeros((6, 1))

    img = preprocessingImage(image)
    
    glcm = graycomatrix(img, [distance], [teta], levels=256, symmetric=True, normed=True)
    
    for i in range(len(indextable[:-1])):
        features = []
        feature = get_feature(glcm, indextable[i])
        features.append(feature)
        data_eye[i, 0] = features[0]
    return pd.DataFrame(np.transpose(data_eye), columns=indextable[:-1])

obj = {
    0.0: "Normal",
    1.0: "Cataract"
}
def predict(image):
    model_rfc = joblib.load("rfc1.pkl")
    model_knn = joblib.load("knn1.pkl")
    model_svm = joblib.load("svm1.pkl")
    model_lr = joblib.load("lr1.pkl")
    model_nb = joblib.load("nb1.pkl")
    X = extract(image)
    results = []
    results.append(obj[model_rfc.predict(X)[0]])
    results.append(obj[model_knn.predict(X)[0]])
    results.append(obj[model_svm.predict(X)[0]])
    # results.append(obj[model_lr.predict(X)[0]])

    normal_count = 0
    cataract_count = 0
    for result in results:
        if (result == 'Normal'):
            normal_count += 1
        else:
            cataract_count+=1
    print(normal_count,cataract_count)        
    if(normal_count > cataract_count):
        actual_result = "Not a Cataract"    
    else:
        actual_result = "Cataract" 
    average_accuracy = (model_rfc.score(X_test,y_test) + model_knn.score(X_test,y_test) + model_svm.score(X_test,y_test) + model_lr.score(X_test,y_test))/4 
    average_accuracy = round(average_accuracy*100,2)              
    st.success('\n\n\n\n\nThe Predicted Label for the image is "{}"\n\n\n\n\n'.format(actual_result))
    st.success("The Accuracy of the Model is {} %".format(average_accuracy))

st.title("CATARACT Disease Prediction")

filename = st.file_uploader('Choose Input Image')


if st.button('Predict'):
    if filename is not None:
        file_bytes = np.asarray(bytearray(filename.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        height,width,channel= opencv_image.shape
        if height == 600 and width == 800 :
            predict(opencv_image)
        else:
            st.warning("Resolution of an Images is in-appropriate \n Make sure that the image you give will have the following dimensions : \n Width : 800 pixels \n Height : 600 pixels")
    else:
        st.warning("Please Select file to proceed")    

