
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt

from sklearn.linear_model import SGDClassifier
import numpy as np

##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
train = 'train.csv'               # path to training file
test = 'test.csv'                 # path to testing file
submission = 'submissionSGD.csv'   # path of to be outputted submission file

# B, parameters
batchReady = False
bsize = 0
batchSize = 10000

MACCCC = 100000

# C, feature/hash trick
D = 2 ** 20             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# B, training/validation
epoch = 1       # learn training data for N passes
holdafter = 29  # data after date N (exclusive) are used as validation
holdout = None  # use every N training instance for holdout validation

##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class basic_SGD(object):
    def __init__(self):
        self.sgd = SGDClassifier(loss="log")

    def predict(self, x, y=None):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # model
        sgd = self.sgd

        if y != None: # training mode
            # should return just p of click = 1
            
            sgd.partial_fit(x, y, [0, 1])
            p = sgd.predict_proba(x)
        else: # testing mode
            p = sgd.predict_proba(x)

        return p[:, 1]

def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for t, row in enumerate(DictReader(open(path))):
        # process id
        ID = row['id']
        del row['id']

        # process clicks
        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']

        # extract date
        date = int(row['hour'][4:6])

        # turn hour really into hour, it was originally YYMMDDHH
        row['hour'] = row['hour'][6:]

        # build x
        x = []
        for key in row:
            value = row[key]

            # one-hot encode everything with hash trick
            index = abs(hash(key + '_' + value)) % D
            x.append(index)

        yield t, date, ID, x, y

##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()
firstTime = True

# initialize ourselves a learner
learner = basic_SGD()

# start training
for e in xrange(epoch):
    loss = 0.
    count = 0

    # since this is just over single sample, either need to update
    # or do in larger batches
    # WARNING: will lose out on left over samples at the end
    for t, date, ID, x, y in data(train, D):  # data is a generator
        #    t: just a instance counter
        # date: you know what this is
        #   ID: id provided in original data
        #    x: features
        #    y: label (click)

        x = np.array(x)
        y = np.array([y])

        # step 1, get prediction from learner
        p = learner.predict(x, y)
        print p

        loss += logloss(p, y)
        count += 1

        if count % 1000 == 0:
            print('Epoch %d finished, validation logloss: %f, elapsed time: %s' % (
            e, loss/count, str(datetime.now() - start)))

        #if count >= MACCCC:
        #	break

    #print('Epoch %d finished, validation logloss: %f, elapsed time: %s' % (
    #    e, loss/count, str(datetime.now() - start)))




##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################

print 'writing test submission...'
with open(submission, 'w') as outfile:
    outfile.write('id,click\n')
    for t, date, ID, x, y in data(test, D):
        p = learner.predict(x)
        outfile.write('%s,%s\n' % (ID, str(p[0])))