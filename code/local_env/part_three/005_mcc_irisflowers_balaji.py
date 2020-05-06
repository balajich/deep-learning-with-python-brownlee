# Accuracy: 88.67%
# Standard deviation: 21.09%
# Multiclass Classification with the Iris Flowers Dataset
import numpy
import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# ------- Begin data preprocessing

# load dataset
dataframe = pandas.read_csv("../data/iris.data", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)
# ------- End data preprocessing

# Define model
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu', kernel_initializer='normal'))
model.add(Dense(3, activation='sigmoid', kernel_initializer='normal'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(X, dummy_y, epochs=150, batch_size=10, validation_split=0.2)
scores = model.evaluate(X, dummy_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
# # define baseline model
# def baseline_model():
#   # create model
#   model = Sequential()
#   model.add(Dense(4, input_dim=4, activation="relu", kernel_initializer="normal"))
#   model.add(Dense(3, kernel_initializer="normal", activation="sigmoid"))
#
#   # Compile model
#   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
#   return model
#
# estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
#
# # Summarize
# print("Accuracy: %.2f%%"%(results.mean()*100))
# print("Standard deviation: %.2f%%"%(results.std()*100))
