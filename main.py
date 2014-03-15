#setup a standard image size; this will distort some images but will get everything into the same shape
import Image
from sklearn.decomposition import RandomizedPCA
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.neighbors import KNeighborsClassifier

STANDARD_SIZE = (300, 167)


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


import numpy as np
import os

img_dirs = ["image/pear/", "image/apple/"]
images = []
for img_dir in img_dirs:
    images += [img_dir + f for f in os.listdir(img_dir)]
import random
random.shuffle(images)
labels = ["pear" if "pear" in f.split('/')[-2] else "apple" for f in images]
print len(filter(lambda x: x == "pear", labels))
print len(filter(lambda x: x == "apple", labels))

data = []
new_labels = []
i = 0
for image, label in zip(images, labels):
    i = i + 1
    try:
        img = img_to_matrix(image)
    except TypeError:
        continue
    except IOError:
        continue
    img = flatten_image(img)
    data.append(img)
    new_labels.append(label)



data = np.array(data)

print new_labels
print len(new_labels)

pca = RandomizedPCA(n_components=2)
X = pca.fit_transform(data)
print len(X[:, 0])
print len(X[:, 1])
df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label": new_labels})
colors = ["red", "yellow"]
for label, color in zip(df['label'].unique(), colors):
    mask = df['label'] == label
    pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
pl.legend()
pl.show()

pca = RandomizedPCA(n_components=5)
test_size = 90
train_x = pca.fit_transform(data[:test_size])
mapped_labels = map(lambda x: 0 if x == "parrot" else 1, new_labels)
test_x = pca.transform(data[test_size:])

print train_x[:5]

knn = KNeighborsClassifier()
knn.fit(train_x, mapped_labels[:test_size])

predicted = knn.predict(test_x)
test_y = mapped_labels[test_size:]
print predicted
print test_y
pd.crosstab(test_y, predicted, rownames=["Actual"], colnames=["Predicted"])