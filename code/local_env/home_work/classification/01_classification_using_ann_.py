from keras.models import Sequential
from keras.layers import Dense
import numpy

import pandas as pd
# fix random seed for reproducibility - it allows that no matter if we execute
# the code more than one time, the random values have to be the same
# Importing the dataset
dataset = pd.read_csv('../../data/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# End of data preprocessing

# create model with hidden layer
model = Sequential()
model.add(Dense(12, input_dim=2, activation="relu", kernel_initializer="uniform"))
model.add(Dense(6, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

# Compile model
# binary_crossentropy = logarithmic loss
# adam = gradient descent algorithm
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=150, batch_size=10)

# Evaluating model with the training data
scores = model.evaluate(X, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
