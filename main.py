#setup a standard image size; this will distort some images but will get everything into the same shape
from PIL import Image
from sklearn.decomposition import RandomizedPCA
import pandas as pd
import matplotlib.pyplot as pl
import cPickle as pickle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os

COLORS = ['y','m','g','c','b','r']
STANDARD_SIZE = (300, 167)
DATA_COUNT = 500 #limiting the number of data

class KNNClassifier:
    def __init__(self, mapping=None):
        self.mapping = mapping

    def predict(self, data):
        test_x = self.pca.transform(data)
        return self.knn.predict(test_x)

    def accuracy(self, data, labels):
        predicted = self.predict(data)
        return np.sum([x == y for (x,y) in zip(predicted, labels)]) / float(len(data))

    def train(self, data, labels, n_components=5, weights='distance', n_neighbors=5):
        self.pca = RandomizedPCA(n_components=n_components)
        fitted = self.pca.fit_transform(data)
        self.knn = KNeighborsClassifier(weights=weights, n_neighbors=n_neighbors)
        self.knn.fit(fitted, labels)

    def visualize(self, data, labels):
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

    def save(self, filename="model.p"):
        pickle.dump({"knn": self.knn, "pca": self.pca, "mapping": self.mapping}, open(filename, 'wb'))

    def load(self, filename="model.p"):
        tmp_dict = pickle.load(open(filename, 'rb'))
        self.knn = tmp_dict["knn"]
        self.pca = tmp_dict["pca"]
        self.mapping = tmp_dict["mapping"]

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




# ONLY FOR VISUALIZATION


# REAL TESTING
#print "==============/nTest start..."
#print len(labels)
#print len(data)
#
## dividing data into train and test sets
#test_size = 10
#
#test_data = data[:test_size]
#test_labels = labels[:test_size]
#
#train_data = data[test_size:]
#train_labels = labels[test_size:]



# testing






def load_data(directory="image"):
    # MAX. 6 CLASSES SUPPORTED (ONLY DUE TO COLORS ON THE PLOT)
    classes = {i: ImgClass(i,name,os.path.join("image",name)) for (i, name) in enumerate(os.listdir("image"))} #class informations

    mapping = {}
    print "=========\nAll data:"
    for cl_id,cl in classes.items():
        print "\nclass name: ",cl.name
        print "class id: ", cl.id
        print "no. of images: ", len(cl.images)
        mapping[id] = cl.name
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

    return data, labels, mapping

def chunk(l, n):
    list = []
    for i in xrange(0, len(l), len(l)/n):
        limit = i+n if i+n < len(l) else len(l)
        list.append(l[i:limit])
    return list

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def cross_validation(input, target, folds=4, **kwargs):
    """
        kwargs are parameters for the model
    """

    zipped = zip(input, target)
    zipped = chunk(zipped, folds)

    accuracy_sum = 0
    for i in range(folds):
        test_zipped = zip(*zipped[i])
        train_zipped = zip(*flatten_list(zipped[:i] + zipped[(i+1):]))
        model = KNNClassifier()
        model.train(train_zipped[0], train_zipped[1])
        accuracy_sum += model.accuracy(test_zipped[0], test_zipped[1])

    return accuracy_sum / float(folds)


def grid_search():
    grid_parameters = {"n_components": [2,6,10], "weights":['distance', 'uniform'], "n_neighbors":[1, 3, 5, 10]}

    data, labels, mapping = load_data()
    print cross_validation(data, labels)

    divisions = []
    operations = 1
    parameter_keys = grid_parameters.keys()
    parameter_values = grid_parameters.values()
    parameter_sizes = map(len, parameter_values)

    for size in parameter_sizes:
        divisions.append(size * operations)
        operations *= size

    print "Iterations: ", operations, " to check"

    for i in range(operations):
        indices = [i % divisions[0]]
        indices += [i / division % division for division in divisions[:-1]]
        i += 1
        grid_iteration_parameters = [pv[index] for (pv, index) in zip(parameter_values, indices)]
        grid_iteration_parameters = dict(zip(parameter_keys, grid_iteration_parameters))
        result = cross_validation(data, labels, **grid_iteration_parameters)
        print "Accuracy: ", result


    model = KNNClassifier(mapping)
    model.train(data, labels)
    model.save()
    model.load()

if __name__ == "__main__":
    #data, labels, mapping = load_data()
    #model = KNNClassifier(mapping)
    #print cross_validation(data, labels)
    #model.train(data, labels)
    #model.save()
    #model.load()
    grid_search()