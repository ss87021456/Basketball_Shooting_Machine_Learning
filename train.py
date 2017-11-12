import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# random shuffled both features and labels 
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# load training data 
print "Loading data..."
features = np.zeros((3200,120))
labels = np.zeros((3200,1))

filepath_1 = 'Features.csv'
f_1 = open(filepath_1, "r")

filepath_2 = 'Labels.csv'
f_2 = open(filepath_2, "r")

for idx,row in enumerate(f_1):
    handle = row.split(',')
    for i in range(120):
        features[idx][i] = float(handle[i])

for idx,row in enumerate(f_2):
    labels[idx] = float(row) 

# Standard Normalization inputs 
print "Data Standard Normalization..."
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)

# Start Training also validation
acc = 0.0
test_time = 5
print "Start Training!"
for i in range(test_time):

    features, labels = unison_shuffled_copies(features, labels)

    train_X = features
    train_y = labels
    test_X = features[-500:]
    test_y = labels[-500:]
    
    # Ready for SVM training

    # first apply pca to reduce dimension from 120 to 5
    n_components = 5
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(train_X)
    projection = np.dot(train_X, pca.components_.T)
    X = projection
    y = train_y
    y = y.reshape([-1])
    kernel = 'rbf'
    
    clf = SVC(kernel=kernel)
    clf.fit(X, y)

    # validation
    projection_test = np.dot(test_X, pca.components_.T)
    predict = (clf.predict(projection_test))
    test_y = test_y.reshape([-1])
    acc += sum(np.array(predict)==np.array(test_y))/float(len(test_y))
    print kernel + " acc:",sum(np.array(predict)==np.array(test_y))/float(len(test_y))

print "Average accuracy of SVC:",acc/test_time * 100,"%"
