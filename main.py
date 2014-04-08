#setup a standard image size; this will distort some images but will get everything into the same shape
from PIL import Image
from sklearn.decomposition import RandomizedPCA
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.neighbors import KNeighborsClassifier

COLORS = ['b','g','r','c','m','y']
STANDARD_SIZE = (300, 167)
DATA_COUNT = 500 #limiting the number of data


def img_to_matrix(filename, verbose=True):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
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

def ImgClass():
    def _init_(self,id,name,path):
        self.id = id
        self.name = name
        self.path = path
        self.images = [Image(id,os.path.join(self.path,name)) for name in os.listdir(self.path)]

def Img():
    def _init_(self,cl_id,path):
        self.cl_id = cl_id
        self.path = path
        self.data = None

    def load(self):
        try:
            data = img_to_matrix(self.path)
            self.data = flatten_image(data)
        except:
            self.data = None

import numpy as np
import os

# MAX. 6 CLASSES SUPPORTED (ONLY DUE TO COLORS ON THE PLOT)
classes = {i: ImageClass(i,name,os.path.join("image",name)) for (i, name) in enumerate(os.listdir("image"))} #class informations

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
    if (img.data):
        data.append(img.data)
        labels.append(img.cl_id)

data = np.array(data)

print "===========\nLoaded data:"
cl_no = 0
for cl_id,cl in classes.items():
    print "\nclass name: ",cl.name
    print "class id: ", cl.id
    l = len(filter(lambda img: img.data,cl.images))
    print "no. of loaded images: ", l
    if (l>0): cl_no+=1
print "\nno. of loaded classes: ",cl_no

# ONLY FOR VISUALIZATION
pca = RandomizedPCA(n_components=2)
X = pca.fit_transform(data)
df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label": new_labels})
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
pca = RandomizedPCA(n_components=5)
test_size = 90
train_x = pca.fit_transform(data[:test_size])
test_x = pca.transform(data[test_size:])

print train_x[:5]

knn = KNeighborsClassifier()
knn.fit(train_x, labels[:test_size])

predicted = knn.predict(test_x)
test_y = labels[test_size:]
print predicted
print test_y
pd.crosstab(test_y, predicted, rownames=["Actual"], colnames=["Predicted"])