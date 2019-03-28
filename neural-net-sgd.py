import tflearn 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("iris.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# convert y into one-hot encoded format
y = y.reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(y)
Y = enc.transform(y).toarray()

net = tflearn.input_data(shape=[None, 4])
net = tflearn.fully_connected(net, 64)
#net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net,128)
net = tflearn.fully_connected(net, 3, activation='sigmoid')
net = tflearn.regression(net, optimizer='sgd', loss='categorical_crossentropy', metric = 'accuracy')

model = tflearn.DNN(net)
model.fit(X, Y, show_metric=True, n_epoch = 500)