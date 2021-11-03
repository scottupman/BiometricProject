''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import cv2
import get_images
import get_landmarks
import numpy as np
import pandas as pd
import performance_plots

from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def getPCA(X):
    # PCA
    npX = np.array(X)
        
    n, nx, ny, nz = npX.shape
    newX = npX.reshape((n,nx*ny*nz))
    
    pca = PCA(n_components=0.8)
    pca.fit(newX)

    dimX = pca.transform(newX)
    X = pca.inverse_transform(dimX)
    
    return X

def getPrePCA(X, preX):
    # PCA
    npX = np.array(X)
        
    n, nx, ny, nz = npX.shape
    newX = npX.reshape((n,nx*ny*nz))
    
    pca = PCA(n_components=0.8)
    pca.fit(newX)
    
    # data to return
    newNPX = np.array(preX)
        
    n, nx, ny = newNPX.shape
    preX = newNPX.reshape((n,nx*ny))

    dimX = pca.transform(preX)
    newX = pca.inverse_transform(dimX)
    
    return newX

def modelA(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    X_train = getPCA(X_train)
    X_test = getPCA(X_test)
    
    accScore = 0
    
    # SVM
    sv = SVC()
    sv.fit(X_train, y_train)
    svPre = sv.predict(X_test)
    accScore += accuracy_score(y_test, svPre) # y_test = true, rfPre = y_pred
    
    
    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rfPre = rf.predict(X_test)
    accScore += accuracy_score(y_test, rfPre) # y_test = true, rfPre = y_pred
    
    # ORC
    orc = ORC(knn())
    orc.fit(X_train, y_train)
    orcPre = orc.predict(X_test)
    accScore += accuracy_score(y_test, orcPre) # y_test = true, rfPre = y_pred

    print("Accuracy: ", accScore/3)    
    
      
def modelB(X, y, XBright, yBright):
    X = getPCA(X)
    XBright = getPCA(XBright)
    
    accScore = 0
    
    # SVM
    sv = SVC()
    sv.fit(X, y)
    svPre = sv.predict(XBright)
    accScore += accuracy_score(yBright, svPre)
    
    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X, y)
    rfPre = sv.predict(XBright)
    accScore += accuracy_score(yBright, rfPre)
    
    # ORC 
    orc = ORC(knn())
    orc.fit(X, y)
    orcPre = orc.predict(XBright)
    accScore += accuracy_score(yBright, orcPre)
    
    print("Accuracy (base vs bright): ", accScore/3)
    
    
    
def modelC(X, y, XDark, yDark):
    X = getPCA(X)
    XDark = getPCA(XDark)
    
    accScore = 0
    
    # SVM
    sv = SVC()
    sv.fit(X, y)
    svPre = sv.predict(XDark)
    accScore += accuracy_score(yDark, svPre)
    
    # Random Forest 
    rf = RandomForestClassifier()
    rf.fit(X, y)
    rfPre = rf.predict(XDark)
    accScore += accuracy_score(yDark, rfPre)
    
    # ORC
    orc = ORC(knn())
    orc.fit(X, y)
    orcPre = orc.predict(XDark)
    accScore += accuracy_score(yDark, orcPre)
    
    print("Accuracy (base vs dark): ", accScore/3)

def modelD(X, y, xBright, yBright):
     # first create grayscale/normalized datasets
    xGray, xNorm, xNoise = [], [], []
    yGray, yNorm, yNoise = yBright, yBright, yBright
    for i in range(len(xBright)):
        xGray.append(cv2.cvtColor(xBright[i], cv2.COLOR_BGR2GRAY))
        xNorm.append(cv2.normalize(xBright[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        xNoise.append(cv2.fastNlMeansDenoisingColored(xBright[i],None, h=10))


    # xBright, yBright = get_landmarks.get_landmarks(xBright, yBright, 'landmarks/', 68, False)
    # xGray, yGray = get_landmarks.get_landmarks(xGray, yGray, 'landmarks/', 68, False)
    # xNorm, yNorm = get_landmarks.get_landmarks(xNorm, yNorm, 'landmarks/', 68, False)
    # xNoise, yNoise = get_landmarks.get_landmarks(xNoise, yNoise, 'landmarks/', 68, False)
    
    accScore = 0
    accScoreNorm = 0
    accScoreNoise = 0
    
    # xGray = getPrePCA(X, xGray)
    X = getPCA(X)
    xBright = getPCA(xBright)
    xNorm = getPCA(xNorm)
    xNoise = getPCA(xNoise)

    # SVC
    svc = SVC()
    svc.fit(X, y)

    # test baseline vs unedited dark images
    svcPre = svc.predict(xBright)
    accScore += accuracy_score(yBright, svcPre)

    # test baseline vs normalized images
    svcPre = svc.predict(xNorm)
    accScoreNorm += accuracy_score(yNorm, svcPre)

    # test baseline vs sobel-filtered images
    svcPre = svc.predict(xNoise)
    accScoreNoise += accuracy_score(yNoise, svcPre)
    #------------------------------------------------
    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X, y)

    # test baseline vs unedited dark images
    rfPre = rf.predict(xBright)
    accScore += accuracy_score(yBright, rfPre)

    # test baseline vs normalized images
    svcPre = svc.predict(xNorm)
    accScoreNorm += accuracy_score(yNorm, rfPre)

    # test baseline vs sobel-filtered images
    svcPre = svc.predict(xNoise)
    accScoreNoise += accuracy_score(yNoise, rfPre)
    #------------------------------------------------
    #ORC
    orc = ORC(knn())
    orc.fit(X, y)

    # test baseline vs unedited dark images
    orcPre = orc.predict(xBright)
    accScore += accuracy_score(yBright, orcPre)
    print("Accuracy Score baseline: ", accScore/3)

    # test baseline vs normalized images
    orcPre = orc.predict(xNorm)
    accScoreNorm += accuracy_score(yNorm, orcPre)
    print("Accuracy Score normalize: ", accScoreNorm/3)


    # test baseline vs sobel-filtered images
    orcPre = orc.predict(xNoise)
    accScoreNoise += accuracy_score(yNoise, orcPre)
    print("Accuracy Score noise: ", accScoreNoise/3)

    

    
# Transferring images for xDark, yDark
def modelE(X, y, xDark, yDark):
    # first create grayscale/normalized datasets
    xGray, xNorm, xNoise = [], [], []
    yGray, yNorm, yNoise = yDark, yDark, yDark
    for i in range(len(xDark)):
        xGray.append(cv2.cvtColor(xDark[i], cv2.COLOR_BGR2GRAY))
        xNorm.append(cv2.normalize(xDark[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        xNoise.append(cv2.fastNlMeansDenoisingColored(xDark[i],None, h=10))
        
          
    # xDark, yDark = get_landmarks.get_landmarks(xDark, yDark, 'landmarks/', 68, False)
    # xGray, yGray = get_landmarks.get_landmarks(xGray, yGray, 'landmarks/', 68, False)
    # xNorm, yNorm = get_landmarks.get_landmarks(xNorm, yNorm, 'landmarks/', 68, False)
    # xNoise, yNoise = get_landmarks.get_landmarks(xNoise, yNoise, 'landmarks/', 68, False)
    
    accScore = 0
    accScoreNorm = 0
    accScoreNoise = 0
    
    X = getPCA(X)
    xDark = getPCA(xDark)
    xNorm = getPCA(xNorm)
    xNoise = getPCA(xNoise)

    # SVM
    svc = SVC()
    svc.fit(X, y)

    # test baseline vs unedited dark images
    svcPre = svc.predict(xDark)
    accScore += accuracy_score(yDark, svcPre)

    # test baseline vs normalized images
    svcPre = svc.predict(xNorm)
    accScoreNorm += accuracy_score(yNorm, svcPre)

    # test baseline vs sobel-filtered images
    svcPre = svc.predict(xNoise)
    accScoreNoise += accuracy_score(yNoise, svcPre)
    #-------------------------------------------------------------
    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X, y)

    # test baseline vs unedited dark images
    rfPre = rf.predict(xDark)
    accScore += accuracy_score(yDark, rfPre)

    # test baseline vs normalized images
    rfPre = rf.predict(xNorm)
    accScoreNorm += accuracy_score(yNorm, rfPre)

    # test baseline vs sobel-filtered images
    rfPre = rf.predict(xNoise)
    accScoreNoise += accuracy_score(yNoise, rfPre)

    #ORC
    orc = ORC(knn())
    orc.fit(X, y)

    # test baseline vs unedited dark images
    orcPre = orc.predict(xDark)
    accScore += accuracy_score(yDark, orcPre)
    print("Accuracy (baseline vs dark, unedited): ", accScore/3)

    # test baseline vs normalized images
    orcPre = orc.predict(xNorm)
    accScoreNorm += accuracy_score(yNorm, orcPre)
    print("Accuracy (baseline vs dark, normalized): ", accScoreNorm/3)

    # test baseline vs sobel-filtered images
    orcPre = orc.predict(xNoise)
    accScoreNoise += accuracy_score(yNoise, orcPre)
    print("Accuracy (baseline vs dark, de-noised): ", accScoreNoise/3)
    #-------------------------------------------------------------    
    
    
''' Import classifier '''
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# NB, SVM, ANN

''' Load the data and their labels '''
image_directory = 'Project 1 Database'
X, y = get_images.get_images(image_directory)
XBright, yBright = get_images.get_bright(image_directory)
XDark, yDark = get_images.get_dark(image_directory)


''' Use PCA on the images '''
modelA(X, y)
modelB(X, y, XBright, yBright)
modelC(X, y, XDark, yDark)
print()
modelD(X, y, XBright, yBright)
print()
modelE(X, y, XDark, yDark)


# ''' Matching and Decision '''
#     # create an instance of the classifier
#     clf = SVC()
    
#     num_correct = 0
#     labels_correct = []
#     num_incorrect = 0
#     labels_incorrect = []
    
#     for i in range(0, len(y)):
#         query_img = X[i, :]
#         query_label = y[i]
        
#         template_imgs = np.delete(X, i, 0)
#         template_labels = np.delete(y, i)
            
#         # Set the appropriate labels
#         # 1 is genuine, 0 is impostor
#         y_hat = np.zeros(len(template_labels))
#         y_hat[template_labels == query_label] = 1 
#         y_hat[template_labels != query_label] = 0
        
#         clf.fit(template_imgs, y_hat) # Train the classifier
#         y_pred = clf.predict(query_img.reshape(1,-1)) # Predict the label of the query
        
#         # Gather results
#         if y_pred == 1:
#             num_correct += 1
#             labels_correct.append(query_label)
#         else:
#             num_incorrect += 1
#             labels_incorrect.append(query_label)

#     # Print results
#     print()
#     print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
#           % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect))) 


 
    
    