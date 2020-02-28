This is a simple implementaion of a KNN classifier which takes two data files as input, one
contain the train set (spam_train.csv) and the other the test set (spam_test.csv).

The data files contain instances of emails which are labeled as 1 for spam and 0 for no-spam.
Each instance has 57 features and the classification is based on an unweighted vote of its
k nearest instances in the training set. Distance measurements are done with Euclideand distance.
