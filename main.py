''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import get_images
import get_landmarks
import numpy as np

''' Import classifier '''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
# NB, SVM, ANN

''' Load the data and their labels '''
image_directory = 'Males'
X, y = get_images.get_images(image_directory)

''' Get distances between face landmarks in the images '''
# get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False)
X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 5, False)

''' Matching and Decision '''
# create an instance of the classifier
clf = NearestCentroid()

num_correct = 0
labels_correct = []
num_incorrect = 0
labels_incorrect = []

for i in range(0, len(y)):
    query_img = X[i, :]
    query_label = y[i]
    
    template_imgs = np.delete(X, i, 0)
    template_labels = np.delete(y, i)
        
    # Set the appropriate labels
    # 1 is genuine, 0 is impostor
    y_hat = np.zeros(len(template_labels))
    y_hat[template_labels == query_label] = 1 
    y_hat[template_labels != query_label] = 0
    
    clf.fit(template_imgs, y_hat) # Train the classifier
    y_pred = clf.predict(query_img.reshape(1,-1)) # Predict the label of the query
    
    # Gather results
    if y_pred == 1:
        num_correct += 1
        labels_correct.append(query_label)
    else:
        num_incorrect += 1
        labels_incorrect.append(query_label)

# Print results
print()
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect)))    
    
    