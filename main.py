''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import get_images
import get_landmarks
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def modelA(X, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # rf = SVC()
    
    # rf.fit(X_train, y_train)
    # rfPre = rf.predict(X_test)
    # accScore = accuracy_score(y_test, rfPre)
    # print("Accuracy: ", accScore)
    
    rf = SVC()
    rf.fit(X, y)
    rfPre = rf.predict(X)
    accScore = accuracy_score(y, rfPre)
    print("Accuracy: ", accScore)
    
    
    
# def modelB():
    
# def modelC():
    
# def modelD():
    
# def modelE():
    

''' Import classifier '''
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# NB, SVM, ANN

''' Load the data and their labels '''
image_directory = 'Project 1 Database'
X, y = get_images.get_images(image_directory)
# xBright, yBright = get_images.get_bright(image_directory)
# xDark, yDark = get_images.get_dark(image_directory)


''' Get distances between face landmarks in the images '''
# get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False)
X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 68, False)

modelA(X, y)



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


 
    
    