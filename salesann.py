import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('Sample - Superstore.csv.xls')

df['Order Date'] = pd.to_datetime(df['Order Date']).dt.date
df['Ship Date'] = pd.to_datetime(df['Ship Date']).dt.date
df.loc[df['Ship Date'].notnull(), 'Days'] = df['Ship Date'] - df['Order Date']

df['Days']=df.apply(lambda row: row.Days.days, axis=1)

arr = df.to_numpy()

X=arr[:,[4,7,12,14,15,17,18,19,20,21]]
y=arr[:,17]

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
labelencoder_X_3 = LabelEncoder()
X[:, 2] = labelencoder_X_3.fit_transform(X[:, 2])
labelencoder_X_4 = LabelEncoder()
X[:, 3] = labelencoder_X_4.fit_transform(X[:, 3])
labelencoder_X_5 = LabelEncoder()
X[:, 4] = labelencoder_X_5.fit_transform(X[:, 4])
transformer = ColumnTransformer(transformers=[("OneHot",OneHotEncoder(),[0,4])],remainder="passthrough")
X = transformer.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = 29))

# Adding the hidden layers
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, validation_split=0.5, epochs=100, batch_size=16, verbose=1)
print(history)
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

