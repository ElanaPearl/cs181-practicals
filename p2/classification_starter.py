## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test directories and parse each xml file into an xml.etree.ElementTree, 
## which is a standard python object used to represent an xml file in memory.
## (More information about xml.etree.ElementTree objects can be found here:
## http://docs.python.org/2/library/xml.etree.elementtree.html
## and here: http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/)
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each ElementTree object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code that randomly picks class-specific weights and predicts
## the class with the weights that give the highest score. If your prediction
## algorithm involves class-specific weights, you should, of course, learn 
## these class-specific weights in a more intelligent way.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take an ElementTree object representing
## an xml file (which contains, among other things, the sequence of system calls a
## piece of potential malware has made), and returns a dictionary mapping feature names to 
## their respective numeric values. 
## For instance, a simple feature-function might map a system call history to the
## dictionary {'first_call-load_image': 1}. This is a boolean feature indicating
## whether the first system call made by the executable was 'load_image'. 
## Real-valued or count-based features can of course also be defined in this way. 
## Because this feature-function will be run over ElementTree objects for each 
## software execution history instance, we will have the (different)
## feature values of this feature for each history, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions (for a particular
## training example) will be unioned, so we can collect all the feature values 
## associated with that particular instance.
##
## Two example feature-functions, first_last_system_call_feats() and 
## system_call_count_feats(), are defined below.
## The first of these functions indicates what the first and last system-calls 
## made by an executable are, and the second records the total number of system
## calls made by an executable.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.
##
## Computational Caveat
## --------------------
## Because the biggest of any of the xml files is only around 35MB, the code below 
## will parse an entire xml file and store it in memory, compute features, and
## then get rid of it before parsing the next one. Storing the biggest of the files 
## in memory should require at most 200MB or so, which should be no problem for
## reasonably modern laptops. If this is too much, however, you can lower the
## memory requirement by using ElementTree.iterparse(), which does parsing in
## a streaming way. See http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/
## for an example. 

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV

import cPickle as pickle

import util


def extract_feats(ffs, direc="train", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      direc is a directory containing xml files (expected to be train or test).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target classes, and a list of system-call-history ids in order 
      of their rows in the design matrix.
      
      Note: the vector of target classes returned will contain the true indices of the
      target classes on the training data, but will contain only -1's on the test
      data
    """
    fds = [] # list of feature dicts
    classes = []
    ids = [] 
    for datafile in os.listdir(direc):
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        rowfd = {}
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        [rowfd.update(ff(tree)) for ff in ffs]
        fds.append(rowfd)
        
    X,feat_dict = make_design_mat(fds,global_feat_dict)
    return X, feat_dict, np.array(classes), ids


def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.
       
    returns: 
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds 
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict
        
    cols = []
    rows = []
    data = []        
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)
   
    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
    return X, feat_dict
    
### Wooooo Features!! #ItsLit
## Here are two example feature-functions. They each take an xml.etree.ElementTree object, 
# (i.e., the result of parsing an xml file) and returns a dictionary mapping 
# feature-names to numeric values.
## TODO: modify these functions, and/or add new ones.
# First and Last system calls of a .xml file
def first_last_system_call_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made. 
      (in other words, it returns a dictionary indicating what the first and 
      last system calls made by an executable were.)
    """
    c = Counter()
    in_all_section = False
    first = True # is this the first system call
    last_call = None # keep track of last call we've seen
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        # start of the list of system calls
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        # end of the list of system calls
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        # within 'all_section'
        elif in_all_section:
            if first:
                c["first_call-"+el.tag] = 1
                first = False
            last_call = el.tag  # update last call seen
            
    # finally, mark last call seen
    c["last_call-"+last_call] = 1
    return c

# Counter with number of occurences for each call in whole .xml
def num_system_call_feats(tree):
    return Counter([i.tag for i in tree.iter()])

# Counter with number of occurrences for each bigram in whole xml
def num_system_call_bifeats(tree):
    # TODO: try only using system calls within all_section
    bi_calls = []
    last_process = None
    for curr_process in tree.iter():
        if last_process:
            bi_calls.append((last_process, curr_process.tag))
        last_process = curr_process.tag
    return Counter(bi_calls)

def num_system_call_trifeats(tree):
    tri_calls = []
    prev_proc = None
    prev_proc2 = None
    for cur_proc in tree.iter():
        if prev_proc2:
            tri_calls.append((prev_proc2, prev_proc, cur_proc.tag))
        if prev_proc:
            prev_proc2 = prev_proc
        prev_proc = curr_proc.tag
    return Counter(tri_calls)

# Counter with number of system calls
def system_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        # end of the list of system calls
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        # within 'all_section'
        elif in_all_section:
            c['num_system_calls'] += 1
    return c

# Counter with number of occurrences of each system call within all_section
def num_sys_call_feats(tree):
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        # start of the list of system calls
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        # end of the list of system calls
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        # within 'all_section'
        elif in_all_section:
            c[el.tag] = c.get(el.tag,0)+1
    return c

# Counter with number of occurrences for each bigram in system
def num_sys_call_bifeats(tree):
    # TODO: try only using system calls within all_section
    in_all_section = False
    bi_calls = []
    last_process = None
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        # end of the list of system calls
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if last_process:
                bi_calls.append((last_process, el.tag))
            last_process = el.tag
    return Counter(bi_calls)

# Counter with number of occurrences for each bigram in system
def num_sys_call_trifeats(tree):
    # TODO: try only using system calls within all_section
    in_all_section = False
    tri_calls = []
    prev_proc = None
    prev_proc2 = None
    for cur_proc in tree.iter():
        # ignore everything outside the "all_section" element
        if cur_proc.tag == "all_section" and not in_all_section:
            in_all_section = True
        # end of the list of system calls
        elif cur_proc.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if prev_proc2:
                tri_calls.append((prev_proc2, prev_proc, cur_proc.tag))
            if prev_proc:
                prev_proc2 = prev_proc
            prev_proc = cur_proc.tag
    return Counter(tri_calls)

# Counter with number of occurrences of each system call
def num_occur_call_feats(tree):
    return Counter([i.tag for i in tree.iter()])

# Counter with number of processes
def num_processes_feats(tree):
    c1 = num_occur_call_feats(tree)
    c2 = {'num_processes': c1['process']}
    return c2

# Counter with number of threads
def num_threads_feats(tree):
    c1 = num_occur_call_feats(tree)
    c2 = {'num_threads': c1['thread']}
    return c2

# Counter with number of tags - estimate for file size
def num_tags_feats(tree):
    c1 = num_occur_call_feats(tree)
    c = {'num_calls': sum(c1.values())}
    return c

def injected_code_feats(tree):
    c=Counter()
    for el in tree.iter():
        if el.tag=='process':
            atr = el.attrib
            if atr['startreason']=='InjectedCode':
                c['num_inject'] = c.get('num_inject',0)+1
    return c

def most_common_feats(tree):
    c1=Counter()
    c=num_sys_call_feats(tree)
    x=sorted(c, key=c.get, reverse=True)
    most_common = x[0]
    c1['most_common-'+most_common]=1
    return c1

## The following function does the feature extraction, learning, and prediction
def main():
    train_dir = "train"
    test_dir = "test"
    outputfile = "mypredictions.csv"  # feel free to change this or take it as an argument
    
    # TODO put the names of the feature functions you've defined above in this list
    # ffs = [first_last_system_call_feats, system_call_count_feats]
    ffs = [first_last_system_call_feats, system_call_count_feats, num_sys_call_feats, num_sys_call_bifeats, num_processes_feats, num_threads_feats, num_tags_feats]
    # ffs = [first_last_system_call_feats, system_call_count_feats, num_sys_call_feats, num_sys_call_bifeats, num_sys_call_trifeats, num_processes_feats, num_threads_feats, num_tags_feats, injected_code_feats, most_common_feats]
    # ffs = [first_last_system_call_feats, system_call_count_feats, num_system_call_feats, num_system_call_bifeats, num_system_call_trifeats, num_tags_feats, injected_code_feats]
    # ffs = [num_sys_call_feats, num_sys_call_bifeats]
    # extract features
    print "extracting training features..."
    X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)

    # Track number of each class
    # spam = Counter()
    # n=0
    # for i in t_train:
    #     spam[i]=spam.get(i,0)+1
    #     n+=1
    # print spam
    # assert(n==3086)

    with open('X_train.pkl','w') as f:
        pickle.dump(X_train, f)

    with open('t_train.pkl','w') as f:
        pickle.dump(t_train, f)

    print "done extracting training features"
    print
    
    # print global_feat_dict

    # with open('X_train_RF_counts.pkl','w') as f:
    #     pickle.dump(X_train, f)


    print "learning..."
    
    # gnb = GaussianNB()
    # gnb.fit(X_train.todense(), t_train)
    rf = RandomForestClassifier(n_estimators=1000, max_features=None)
    rf.fit(X_train.todense(), t_train)

    print "done learning"
    print

    # cross validation
    print "cross validation..."
    scores = cross_val_score(gnb, X_train.todense(), t_train, cv=5)
    # scores = cross_val_score(rf, X_train.todense(), t_train, cv=5)
    print scores
    print

    # Grid Search CV
    # print "grid search..."
    # RF = RandomForestClassifier(max_features=None)
    # RF_params = {'n_estimators':[100,1000,2000,5000,10000]}
    # gsc = GridSearchCV(estimator=RF, param_grid=RF_params, cv=5)
    # gsc.fit(X_train.todense(), t_train)
    # print "grid search done"
    # print "Best score:", gsc.best_score_
    # print "Best estimator:", gsc.best_estimator_
    # print

    # get rid of training data and load test data
    del X_train
    del t_train
    del train_ids
    print "extracting test features..."
    X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
    print "done extracting test features"
    # # with open('X_test_RF_counts.pkl','w') as f:
    # #     pickle.dump(X_test, f)

    with open('X_test.pkl','w') as f:
        pickle.dump(X_test, f)
    
    with open('test_ids.pkl','w') as f:
        pickle.dump(test_ids, f)
    print

    # Measure accuracy on each class in the training data
    # print "testing on train data..."
    # pred_train = rf.predict(X_train.todense())
    # print pred_train
    # print t_train
    # spam_pred = Counter()
    # n1=0
    # for i in range(n):
    #     c=t_train[i]
    #     if c==pred_train[i]:
    #         spam_pred[c]=spam_pred.get(c,0)+1
    #         n1+=1
    # print spam_pred
    # for i in spam_pred.keys():
    #     num_pred = spam_pred[i]
    #     num_act = spam[i]
    #     acc = float(num_pred)/num_act
    #     print "Percentage of class",i,"classified correctly:", "{0:.3f}".format(acc),"(",num_pred,"out of",num_act,")"
    # overall_acc = float(n1)/n
    # print "Overall accuracy:", "{0:.3f}".format(overall_acc)
    # print

    print "making predictions..."
    # preds = gnb.predict(X_test.todense())
    preds = rf.predict(X_test.todense())
    # preds = gsc.predict(X_test.todense())    

    print "done making predictions"
    print
    
    print "writing predictions..."
    util.write_predictions(preds, test_ids, outputfile)
    print "done!"

if __name__ == "__main__":
    main()
    