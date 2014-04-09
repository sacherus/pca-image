#setup a standard image size; this will distort some images but will get everything into the same shape
from PIL import Image
from sklearn.decomposition import RandomizedPCA
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.neighbors import KNeighborsClassifier

COLORS = ['y','m','g','c','b','r']
STANDARD_SIZE = (300, 167)
DATA_COUNT = 500 #limiting the number of data


def img_to_matrix(filename, verbose=True):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    # print img.size
    if verbose:
        print "changing size from %s to %s in %s" % (str(img.size), str(STANDARD_SIZE), filename)
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img


def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)

    return img_wide[0]

class ImgClass:
    def __init__(self,id,name,path):
        self.id = id
        self.name = name
        self.path = path
        self.images = [Img(id,os.path.join(self.path,name)) for name in os.listdir(self.path)]

class Img:
    def __init__(self,cl_id,path):
        self.cl_id = cl_id
        self.path = path
        self.data = None

    def load(self):
        try:
            data = img_to_matrix(self.path,verbose=False)
            self.data = flatten_image(data)
            if(len(self.data)!=150300): raise Error
        except:
            self.data = None

import numpy as np
import os

# MAX. 6 CLASSES SUPPORTED (ONLY DUE TO COLORS ON THE PLOT)
classes = {i: ImgClass(i,name,os.path.join("image",name)) for (i, name) in enumerate(os.listdir("image"))} #class informations

print "=========\nAll data:"
for cl_id,cl in classes.items():
    print "\nclass name: ",cl.name
    print "class id: ", cl.id
    print "no. of images: ", len(cl.images)
print "\ntotal no. of classes: ",len(classes)

images = []
for cl_id,cl in classes.items():
    images = images + cl.images

import random
random.shuffle(images)

#optional limiting the number of data
#images = images[:DATA_COUNT]

data = []
labels = []

for img in images:
    img.load()
    if (img.data != None):
        data.append(img.data)
        labels.append(img.cl_id)

data = np.array(data)

print "===========\nLoaded data:"
cl_no = 0
for cl_id,cl in classes.items():
    print "\nclass name: ",cl.name
    print "class id: ", cl.id
    l = len(filter(lambda img: img.data != None,cl.images))
    print "no. of loaded images: ", l
    if (l>0): cl_no+=1
print "\nno. of loaded classes: ",cl_no

# ONLY FOR VISUALIZATION
pca = RandomizedPCA(n_components=2)
X = pca.fit_transform(data)
df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label": labels})
try:
    colors = COLORS[:cl_no]
except IndexError:
    print "Error: too many classes to plot. Make sure there are not more than 6 subdirectories in the 'image/' directory"
for label, color in zip(df['label'].unique(), colors):
    mask = df['label'] == label
    pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
pl.legend()
pl.show()

# REAL TESTING
print "==============/nTest start..."
print len(labels)
print len(data)
pca = RandomizedPCA(n_components=5)

# dividing data into train and test sets
test_size = 10

test_data = data[:test_size]
test_labels = labels[:test_size]

train_data = data[test_size:]
train_labels = labels[test_size:]

# training PCA and KNN
train_x = pca.fit_transform(train_data)
knn = KNeighborsClassifier()
knn.fit(train_x, train_labels)

# testing
test_x = pca.transform(test_data)
predicted = knn.predict(test_x)

test_labels =np.array(test_labels)

#results
print predicted
print test_labels
# u mnie ponizsza linijka nic nie wypisuje - cos trzeba z tym jeszcze zrobic?
pd.crosstab(test_labels, predicted, rownames=["Actual"], colnames=["Predicted"])
