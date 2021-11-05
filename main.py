''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import cv2
import get_images
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
    
    # SVM
    sv = SVC(probability = True)
    sv.fit(X_train, y_train)
    matching_scores_sv = sv.predict_proba(X_test)
    
    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    matching_scores_rf = rf.predict_proba(X_test)
    
    # ORC
    orc = ORC(knn())
    orc.fit(X_train, y_train)
    matching_scores_orc = orc.predict_proba(X_test)
    
    # Score fusion with matching scores
    matching_scores = (matching_scores_orc + matching_scores_rf + matching_scores_sv) / 3.0

    gen_scores = []
    imp_scores = []
    classes = orc.classes_
    matching_scores = pd.DataFrame(matching_scores, columns = classes)
    
    for i in range(len(y_test)):
        scores = matching_scores.loc[i]
        mask = scores.index.isin([y_test[i]])
        gen_scores.extend(scores[mask])
        imp_scores.extend(scores[~mask])
        
    performance_plots.performance(gen_scores, imp_scores, 'rf-orc-svm-score fusion for baseline', 100)
    
      
def modelB(X, y, XBright, yBright):
    X = getPCA(X)
    XBright = getPCA(XBright)
    
    # SVM
    sv = SVC(probability = True)
    sv.fit(X, y)
    matching_scores_sv = sv.predict_proba(XBright)
    
    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X, y)
    matching_scores_rf = rf.predict_proba(XBright)
    
    # ORC 
    orc = ORC(knn())
    orc.fit(X, y)
    matching_scores_orc = orc.predict_proba(XBright)
        
    # Score fusion with matching scores
    matching_scores = (matching_scores_orc + matching_scores_rf + matching_scores_sv) / 3.0
    
    gen_scores = []
    imp_scores = []
    classes = orc.classes_
    matching_scores = pd.DataFrame(matching_scores, columns = classes)
    
    for i in range(len(yBright)):
        scores = matching_scores.loc[i]
        mask = scores.index.isin([yBright[i]])
        gen_scores.extend(scores[mask])
        imp_scores.extend(scores[~mask])
        
    performance_plots.performance(gen_scores, imp_scores, 'rf-orc-svm-score fusion for bright', 100)
    
    
def modelC(X, y, XDark, yDark):
    X = getPCA(X)
    XDark = getPCA(XDark)
    
    # SVM
    sv = SVC(probability = True)
    sv.fit(X, y)
    matching_scores_sv = sv.predict_proba(XDark)
    
    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X, y)
    matching_scores_rf = rf.predict_proba(XDark)
    
    # ORC 
    orc = ORC(knn())
    orc.fit(X, y)
    matching_scores_orc = orc.predict_proba(XDark)
        
    # Score fusion with matching scores
    matching_scores = (matching_scores_orc + matching_scores_rf + matching_scores_sv) / 3.0
    
    gen_scores = []
    imp_scores = []
    classes = orc.classes_
    matching_scores = pd.DataFrame(matching_scores, columns = classes)
    
    for i in range(len(yDark)):
        scores = matching_scores.loc[i]
        mask = scores.index.isin([yDark[i]])
        gen_scores.extend(scores[mask])
        imp_scores.extend(scores[~mask])
        
    performance_plots.performance(gen_scores, imp_scores, 'rf-orc-svm-score fusion for dark', 100)

def modelD(X, y, xBright, yBright):
    xContrast, xNorm, xNoise = [], [], []
    yContrast, yNorm, yNoise = yBright, yBright, yBright
    for i in range(len(xBright)):
        imgSrc = xBright[i]
        r_image, g_image, b_image = cv2.split(imgSrc)
        
        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)
        
        xContrast.append(cv2.merge((r_image_eq, g_image_eq, b_image_eq)))
                      
        xNorm.append(cv2.normalize(xBright[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        
        xNoise.append(cv2.fastNlMeansDenoisingColored(xBright[i],None, h=10))

    X = getPCA(X)
    xContrast = getPCA(xContrast)
    xNorm = getPCA(xNorm)
    xNoise = getPCA(xNoise)

    # SVC
    svc = SVC(probability=(True))
    svc.fit(X, y)
    
    # test baseline vs contrast-equalized
    matching_scores_svc_contrast = svc.predict_proba(xContrast)

    # test baseline vs normalized images
    matching_scores_svc_norm = svc.predict_proba(xNorm)

    # test baseline vs sobel-filtered images
    matching_scores_svc_noise = svc.predict_proba(xNoise)
    
    #------------------------------------------------
    
    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X, y)

    # test baseline vs contrast-equalized
    matching_scores_rf_contrast = svc.predict_proba(xContrast)

    # test baseline vs normalized images
    matching_scores_rf_norm = rf.predict_proba(xNorm)

    # test baseline vs noise-filtered
    matching_scores_rf_noise = rf.predict_proba(xNoise)
    
    #------------------------------------------------
    
    #ORC
    orc = ORC(knn())
    orc.fit(X, y)
    
    # test baseline vs contrast-equalized
    matching_scores_orc_contrast = svc.predict_proba(xContrast)
    
    # test baseline vs normalized images
    matching_scores_orc_norm = orc.predict_proba(xNorm)


    # test baseline vs noise-filtered
    matching_scores_orc_noise = orc.predict_proba(xNoise)
    
    # Score fusion with matching scores
    matching_scores_contrast = (matching_scores_orc_contrast + matching_scores_rf_contrast + matching_scores_svc_contrast) / 3.0
    matching_scores_norm = (matching_scores_orc_norm + matching_scores_rf_norm + matching_scores_svc_norm) / 3.0
    matching_scores_noise = (matching_scores_orc_noise + matching_scores_rf_noise + matching_scores_svc_noise) / 3.0
    
    # Gen and impostor for contrast
    gen_scores_contrast = []
    imp_scores_contrast = []
    classes = orc.classes_
    matching_scores_contrast = pd.DataFrame(matching_scores_contrast, columns = classes)

    for i in range(len(yBright)):
        scores = matching_scores_contrast.loc[i]
        mask = scores.index.isin([yBright[i]])
        gen_scores_contrast.extend(scores[mask])
        imp_scores_contrast.extend(scores[~mask])

    performance_plots.performance(gen_scores_contrast, imp_scores_contrast, 'rf-orc-svm-score fusion for contrast bright', 100)
    
    # Gen and impostor for norm
    gen_scores_norm = []
    imp_scores_norm = []
    classes = orc.classes_
    matching_scores_norm = pd.DataFrame(matching_scores_norm, columns = classes)

    for i in range(len(yBright)):
        scores = matching_scores_norm.loc[i]
        mask = scores.index.isin([yBright[i]])
        gen_scores_norm.extend(scores[mask])
        imp_scores_norm.extend(scores[~mask])

    performance_plots.performance(gen_scores_norm, imp_scores_norm, 'rf-orc-svm-score fusion for norm bright', 100)
    
    # Gen and impostor for noise
    gen_scores_noise = []
    imp_scores_noise = []
    classes = orc.classes_
    matching_scores_noise = pd.DataFrame(matching_scores_noise, columns = classes)

    for i in range(len(yBright)):
        scores = matching_scores_noise.loc[i]
        mask = scores.index.isin([yBright[i]])
        gen_scores_noise.extend(scores[mask])
        imp_scores_noise.extend(scores[~mask])

    performance_plots.performance(gen_scores_noise, imp_scores_noise, 'rf-orc-svm-score fusion for noise bright', 100)
    

    
# Transferring images for xDark, yDark
def modelE(X, y, xDark, yDark):   
    xContrast, xNorm, xNoise = [], [], []
    yContrast, yNorm, yNoise = yDark, yDark, yDark
    for i in range(len(xDark)):     
        imgSrc = xDark[i]
        r_image, g_image, b_image = cv2.split(imgSrc)
        
        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)
        
        xContrast.append(cv2.merge((r_image_eq, g_image_eq, b_image_eq)))
                      
        xNorm.append(cv2.normalize(xDark[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        
        xNoise.append(cv2.fastNlMeansDenoisingColored(xDark[i],None, h=10))
        
    X = getPCA(X)
    xContrast = getPCA(xContrast)
    xNorm = getPCA(xNorm)
    xNoise = getPCA(xNoise)

    # SVC
    svc = SVC(probability=(True))
    svc.fit(X, y)
    
    # test baseline vs contrast-equalized
    matching_scores_svc_contrast = svc.predict_proba(xContrast)

    # test baseline vs normalized images
    matching_scores_svc_norm = svc.predict_proba(xNorm)

    # test baseline vs sobel-filtered images
    matching_scores_svc_noise = svc.predict_proba(xNoise)
    
    #------------------------------------------------
    
    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X, y)

    # test baseline vs contrast-equalized
    matching_scores_rf_contrast = svc.predict_proba(xContrast)

    # test baseline vs normalized images
    matching_scores_rf_norm = rf.predict_proba(xNorm)

    # test baseline vs noise-filtered
    matching_scores_rf_noise = rf.predict_proba(xNoise)
    
    #------------------------------------------------
    
    #ORC
    orc = ORC(knn())
    orc.fit(X, y)
    
    # test baseline vs contrast-equalized
    matching_scores_orc_contrast = svc.predict_proba(xContrast)
    
    # test baseline vs normalized images
    matching_scores_orc_norm = orc.predict_proba(xNorm)


    # test baseline vs noise-filtered images
    matching_scores_orc_noise = orc.predict_proba(xNoise)
    
    # Score fusion with matching scores
    matching_scores_contrast = (matching_scores_orc_contrast + matching_scores_rf_contrast + matching_scores_svc_contrast) / 3.0
    matching_scores_norm = (matching_scores_orc_norm + matching_scores_rf_norm + matching_scores_svc_norm) / 3.0
    matching_scores_noise = (matching_scores_orc_noise + matching_scores_rf_noise + matching_scores_svc_noise) / 3.0
    
    # Gen and impostor for contrast
    gen_scores_contrast = []
    imp_scores_contrast = []
    classes = orc.classes_
    matching_scores_contrast = pd.DataFrame(matching_scores_contrast, columns = classes)

    for i in range(len(yBright)):
        scores = matching_scores_contrast.loc[i]
        mask = scores.index.isin([yBright[i]])
        gen_scores_contrast.extend(scores[mask])
        imp_scores_contrast.extend(scores[~mask])

    performance_plots.performance(gen_scores_contrast, imp_scores_contrast, 'rf-orc-svm-score fusion for contrast dark', 100)
    
    # Gen and impostor for norm
    gen_scores_norm = []
    imp_scores_norm = []
    classes = orc.classes_
    matching_scores_norm = pd.DataFrame(matching_scores_norm, columns = classes)

    for i in range(len(yBright)):
        scores = matching_scores_norm.loc[i]
        mask = scores.index.isin([yBright[i]])
        gen_scores_norm.extend(scores[mask])
        imp_scores_norm.extend(scores[~mask])

    performance_plots.performance(gen_scores_norm, imp_scores_norm, 'rf-orc-svm-score fusion for norm dark', 100)
    
    # Gen and impostor for noise
    gen_scores_noise = []
    imp_scores_noise = []
    classes = orc.classes_
    matching_scores_noise = pd.DataFrame(matching_scores_noise, columns = classes)

    for i in range(len(yBright)):
        scores = matching_scores_noise.loc[i]
        mask = scores.index.isin([yBright[i]])
        gen_scores_noise.extend(scores[mask])
        imp_scores_noise.extend(scores[~mask])

    performance_plots.performance(gen_scores_noise, imp_scores_noise, 'rf-orc-svm-score fusion for noise dark', 100)  
    
    
''' Load the data and their labels '''
image_directory = 'Project 1 Database'
X, y = get_images.get_images(image_directory)
XBright, yBright = get_images.get_bright(image_directory)
XDark, yDark = get_images.get_dark(image_directory)


''' Use PCA on the images '''
modelA(X, y)
modelB(X, y, XBright, yBright)
modelC(X, y, XDark, yDark)
modelD(X, y, XBright, yBright)
modelE(X, y, XDark, yDark)



 
    
    