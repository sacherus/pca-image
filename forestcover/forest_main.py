import numpy
import cPickle as pickle
import math
import random

from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.structure.modules import SoftmaxLayer, SigmoidLayer, TanhLayer
import time
from sklearn import preprocessing
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score


train_filename = "forestcover.train"
test_filename = "forestcover.test"
validation_filename = "forestcover.valid"
load_cache = True
save_cache = False
normalization = True
#how many examples i want read
#examples = float(1000)
examples = float("inf")


#don't change!
attributes = 54
classes_number = 7

import sys
import codecs

sys.stdout = codecs.getwriter('utf8')(sys.stdout)


class FNNClassifier:
    model = None

    def __init__(self, scaler=None):
        self.scaler = scaler

    def predict(self, tstdata):
        """returns out, target

        """
        return self.model.testOnClassData(dataset=tstdata, return_targets=True, verbose=False)

    def train(self, trndata, valdata, hidden_neurons=5, hidden_class=SigmoidLayer, iterations=3):
        print "Hidden neurons: " + str(hidden_neurons)
        print "Hidden class: " + str(hidden_class)
        print "Iterations: " + str(iterations)

        fnn = buildNetwork(trndata.indim, hidden_neurons, trndata.outdim, outclass=SoftmaxLayer,
                           hiddenclass=hidden_class)
        trainer = RPropMinusTrainer(fnn, dataset=trndata, verbose=False)
        #trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.5, verbose=True, learningrate=0.05)

        for i in range(iterations):
            trainer.train()
            out, tar = trainer.testOnClassData(dataset=valdata, return_targets=True, verbose=False)
            #used to return final score, not used yet :D
            print str(i) + " " + str(accuracy(out, tar))

        self.model = trainer

    def save(self, filename):
        pickle.dump({"model": self.model, "scaler": self.scaler}, open(filename, 'wb'))

    def load(self, filename):
        my_dict = pickle.load(open(filename, "rb"))
        self.model = my_dict["model"]
        self.scaler = my_dict["scaler"]

def st_time(func):
    """
        st decorator to calculate the total time of a func
    """

    def st_func(*args, **keyArgs):
        t1 = time.time()
        r = func(*args, **keyArgs)
        t2 = time.time()
        print func.__name__, t2 - t1
        return r

    return st_func


def read_configuration():
    """
        reads configuration for podolak data type
    """
    filename = "forestcover.conf"
    tmp_dict = {}
    with open(filename) as f:
        for line in f:
            data = line.strip().split("=")
            tmp_dict[data[0].strip()] = data[1].strip()

    return tmp_dict

@st_time
def read_data_file(filename):
    """
    reads file name filename all examples or number set in examples variable
    """
    attributes = 54
    classes_number = 7
    all_data = ClassificationDataSet(attributes, nb_classes=classes_number)
    with open(filename) as f:
        i = 0
        for line in f:
            i = i + 1
            splited_line = map(lambda x: int(x), line.strip().split(","))
            all_data.addSample(splited_line[:-1], splited_line[-1] - 1)
            if i > examples:
                break
    return all_data


@st_time
def my_dump(*args, **keyArgs):
    return pickle.dump(*args, **keyArgs)


@st_time
def my_load(filename):
    return pickle.load((open(filename + ".p", "rb")))


def normalize_2_sets(trndata, tstdata):
    """scaler for data scaling, don't forget to standarize training set (don't confuse with validating

    taking 2 sets and norma
    """
    trn = trndata.data['input'][:, :10]
    tst = tstdata['input'][:, :10]
    stacked = np.vstack((trn, tst))
    #don't know which set should i userd here, stacked or trn, betters scoers for validation with stacked
    scaler = preprocessing.StandardScaler().fit(stacked)
    scaler.transform(trn, copy=False)
    scaler.transform(tst, copy=False)
    return scaler


def load_data():
    trndata = None
    tstdata = None
    scaler = None
    if not load_cache:
        trndata = read_data_file(train_filename)
        valdata, trndata = trndata.splitWithProportion(0.35)
        tstdata = read_data_file(test_filename)

        if normalization:
            scaler = normalize_2_sets(trndata, tstdata)
        trndata._convertToOneOfMany()
        tstdata._convertToOneOfMany()

    if save_cache:
        my_dump(trndata, open(train_filename + ".p", "wb"))
        my_dump(tstdata, open(test_filename + ".p", "wb"))
        my_dump(valdata, open(validation_filename + ".p", "wb"))
        my_dump(scaler, open("scaler" + ".p", "wb"))

    if load_cache and not save_cache:
        trndata = my_load(train_filename)
        valdata = my_load(validation_filename)
        tstdata = my_load(test_filename)

    print trndata.calculateStatistics()
    print valdata.calculateStatistics()

    return (trndata, valdata, tstdata)


def accuracy(x, y):
    """
    calculating accuracy, can write in better way
    """
    hit = 0
    for a, b in zip(x, y):
        if a == b: hit += 1
    return float(hit) / len(x)


#def misclassification(x, y):
#    """
#        could be error measure
#    """
#    return 1 - accuracy(x, y)


def score_model(trndata, valdata, tstdata, scoring, **kwargs):
    ann = FNNClassifier()
    ann.train(trndata, valdata, **kwargs)
    true_labels, prediction_labels = ann.predict(tstdata)
    return scoring(true_labels, prediction_labels)


def confusion_matrix(y_true, y_pred, chosen_class=0):
    tp = sum(yt == yp and chosen_class == yt for (yt, yp) in zip(y_true, y_pred))
    fp = sum(yt != yp and chosen_class == yp for (yt, yp) in zip(y_true, y_pred))
    tn = sum(chosen_class != yt and chosen_class != yp for (yt, yp) in zip(y_true, y_pred))
    fn = sum(chosen_class == yt and chosen_class != yp for (yt, yp) in zip(y_true, y_pred))
    return {"tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn}


def mcc(y_true, y_pred, chosen_class=0):
    cm = confusion_matrix(y_true, y_pred, chosen_class)
    nominator = cm['tp'] * cm['tn'] - cm['fp'] * cm['fn']
    print cm
    denominator = math.sqrt((cm['tp'] + cm['fp']) * (cm['tp'] + cm['fn']) * (cm['tn'] + cm['fp']) * (cm['tn'] + cm['fn']))
    if denominator == 0:
        print "Equal 0"
        return 0
    return nominator / float(denominator)


@st_time
def main():
    grid_parameters = {"hidden_class": [SigmoidLayer, TanhLayer], "hidden_neurons": [120, 180, 240, 300],
                       "iterations": [10, 25, 40]}
    (trndata, valdata, tstdata) = load_data()

    divisions = []
    operations = 1
    parameter_keys = grid_parameters.keys()
    parameter_values = grid_parameters.values()
    parameter_sizes = map(len, parameter_values)

    for size in parameter_sizes:
        divisions.append(size * operations)
        operations *= size

    print divisions

    print operations

    for i in range(operations):
        indices = [i % divisions[0]]
        indices += [i / division % division for division in divisions[:-1]]
        i += 1
        grid_iteration_parameters = [pv[index] for (pv, index) in zip(parameter_values, indices)]
        grid_iteration_parameters = dict(zip(parameter_keys, grid_iteration_parameters))
        result = score_model(trndata, valdata, tstdata, accuracy, **grid_iteration_parameters)
        print "Test set: " + str(result)




def cross_validation(trndata, folds=3, **kwargs):
    """
        kwargs are parameters for the model
    """
    input = np.vsplit(trndata['input'], folds)
    target = np.vsplit(trndata['target'], folds)

    zipped = zip(input, target)

    accuracy_sum = 0
    for i in len(zipped):
        new_train = ClassificationDataSet(attributes, nb_classes=classes_number)
        new_test = ClassificationDataSet(attributes, nb_classes=classes_number)
        test_zipped = zipped[i]
        train_zipped = zipped[:i] + zipped[(i+1):]

        new_train.setField('input', np.vstack[train_zipped[0]])
        new_train.setField('target', np.vstack[train_zipped[1]])

        new_test.setField('input', test_zipped[0])
        new_test.setField('target', train_zipped[1])

        model = FNNClassifier()
        model.train(new_train, new_test, kwargs)
        out, targ = model.predict(new_test)
        accuracy_sum += accuracy(out, targ)

    return accuracy_sum / folds

def bootstrap(trndata, iter=100):
    """
    check http://sci2s.ugr.es/keel/pdf/specific/articulo/jain_boot_87.pdf for notation
    """
    print trndata.calculateStatistics()
    np_array = np.hstack((trndata['input'], trndata['target']))
    my_range = range(np_array.shape[0])

    print trndata['target'].shape

    app_sum = 0
    e0_sum = 0
    for i in range(iter):
        indices = list(set([random.choice(my_range) for i in my_range]))
        np_train_array = np.vstack(np_array[indices])
        new_training_samples = ClassificationDataSet(attributes, classes_number)
        new_training_samples.setField('input', np_train_array[:, :54])
        new_training_samples.setField('target', np_train_array[:, 54:55])
        new_training_samples._convertToOneOfMany()

        test_indices = list(set(my_range) - set(indices))
        new_test_samples = ClassificationDataSet(attributes, classes_number)
        np_test_array = np.vstack(np_array[test_indices])

        new_test_samples.setField('input', np_test_array[:, :54])
        new_test_samples.setField('target', np_test_array[:, 54:55])
        new_test_samples._convertToOneOfMany()

        print new_training_samples.calculateStatistics()
        print new_test_samples.calculateStatistics()

        model = FNNClassifier()
        model.train(new_training_samples, new_test_samples)

        (xtrn, ytrn) = model.predict(new_training_samples)
        (xtest, ytest) = model.predict(new_test_samples)

        app_sum += (1 - accuracy(xtrn, ytrn))
        e0_sum += (1 - accuracy(xtest, ytest))

    app = app_sum / float(iter)
    e0 = e0_sum / float(iter)

    e632 = 0.368 * app + 0.632 * e0

    print e632
    return e632


def predict():
    config = read_configuration()
    data_set = read_data_file(config["data"])
    model = FNNClassifier()
    model.load(config["model"])
    model.scaler.transform(data_set['input'][:, :10], copy=False)
    data_set._convertToOneOfMany()
    predicted, target =  model.predict(data_set)
    if config["error"] == "accuracy":
        acc = accuracy(predicted, target)
        print "Accuracy: " + str(accuracy(predicted, target))
        print "Error: " + str(1 - acc)
    else:
        mccs = [mcc(target, predicted, my_class) for my_class in range(classes_number)]
        avg_mcc = sum(mccs) / classes_number
        print "MCC per class " + str(mccs)
        print "avg MCC: " + str(avg_mcc)
        print "Error: What is error in mcc?"
    f = open(config["outfile"], 'w')
    f.write("\n".join(map(lambda x: str(x+1), predicted)))
    f.close()


def learn():
    model_filename = "mymodel.p"
    data_train = read_data_file(train_filename)
    data_test = read_data_file(test_filename)
    data_train._convertToOneOfMany()
    data_test._convertToOneOfMany()
    scaler = normalize_2_sets(data_train, data_test)
    model = FNNClassifier(scaler)
    model.train(data_train, data_test)
    model.save(model_filename)



if __name__ == "__main__":
    predict()
