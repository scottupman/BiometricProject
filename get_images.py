# Load imports
import os 
import cv2

'''
function get_images() definition
Parameter, image_directory, is the directory 
holding the images
'''

def get_images(image_directory):
    X = []
    y = []
    extensions = ('jpg','png','gif')
    fileNames = ('BL_BL_1', 'BL_BL_2', 'BL_BL_3', 'BL_BL_4', 'BL_BL_5')
    
    '''
    Each subject has their own folder with their
    images. The following line lists the names
    of the subfolders within image_directory.
    '''
    subfolders = os.listdir(image_directory)
    for subfolder in subfolders:
        print("Loading images in %s" % subfolder)
        if os.path.isdir(os.path.join(image_directory, subfolder)): # only load directories
            subfolder_files = os.listdir(
                    os.path.join(image_directory, subfolder)
                    )
            for file in subfolder_files:
                if file.endswith(extensions): # grab images only
                    if file.startswith(fileNames):
                        # read the image using openCV                    
                        img = cv2.imread(
                                os.path.join(image_directory, subfolder, file)
                                )
                        # resize the image                     
                        width = 100
                        height = 100
                        dim = (width, height)
                        img = cv2.resize(img, dim)
                        
                        # add the resized image to a list X
                        X.append(img)
                        # add the image's label to a list y
                        y.append(subfolder)
                        

    
    print("All images are loaded")     
    # return the images and their labels      
    return X, y
                  
def get_bright(image_directory):
    X = []
    y = []
    extensions = ('jpg','png','gif')
    fileNames = ('L_B_1', 'L_B_2')
    
    '''
    Each subject has their own folder with their
    images. The following line lists the names
    of the subfolders within image_directory.
    '''
    subfolders = os.listdir(image_directory)
    for subfolder in subfolders:
        print("Loading images in %s" % subfolder)
        if os.path.isdir(os.path.join(image_directory, subfolder)): # only load directories
            subfolder_files = os.listdir(
                    os.path.join(image_directory, subfolder)
                    )
            for file in subfolder_files:
                if file.endswith(extensions): # grab images only
                    if file.startswith(fileNames):
                        # read the image using openCV                    
                        img = cv2.imread(
                                os.path.join(image_directory, subfolder, file)
                                )
                        # resize the image                     
                        width = 100
                        height = 100
                        dim = (width, height)
                        img = cv2.resize(img, dim)
                        
                        # add the resized image to a list X
                        X.append(img)
                        # add the image's label to a list y
                        y.append(subfolder)
                        

    
    print("All images are loaded")     
    # return the images and their labels      
    return X, y

def get_dark(image_directory):
    X = []
    y = []
    extensions = ('jpg','png','gif')
    fileNames = ('L_D_1', 'L_D_2')
    
    '''
    Each subject has their own folder with their
    images. The following line lists the names
    of the subfolders within image_directory.
    '''
    subfolders = os.listdir(image_directory)
    for subfolder in subfolders:
        print("Loading images in %s" % subfolder)
        if os.path.isdir(os.path.join(image_directory, subfolder)): # only load directories
            subfolder_files = os.listdir(
                    os.path.join(image_directory, subfolder)
                    )
            for file in subfolder_files:
                if file.endswith(extensions): # grab images only
                    if file.startswith(fileNames):
                        # read the image using openCV                    
                        img = cv2.imread(
                                os.path.join(image_directory, subfolder, file)
                                )
                        # resize the image                     
                        width = 100
                        height = 100
                        dim = (width, height)
                        img = cv2.resize(img, dim)
                        
                        # add the resized image to a list X
                        X.append(img)
                        # add the image's label to a list y
                        y.append(subfolder)
                        

    
    print("All images are loaded")     
    # return the images and their labels      
    return X, y
    

                
            