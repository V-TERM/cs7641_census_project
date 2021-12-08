from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from keras import backend as K
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

# 'pres' or 'sen'
CLASS = "sen"

class CensusRNN:
	def __init__(self, batch_size=128, no_features=175) -> None:
		self.BATCH_SIZE = batch_size
		self.LEN_FEATURES = no_features

		self.model = None

	def load_dataset(self):
	 
		# Load the dataset
		self.data = np.load(f"X_{CLASS}.npy")
		self.labels = np.load(f"y_{CLASS}.npy")
  
		print(f"Loaded {CLASS} dataset")
		print(f"Shape of data: {self.data.shape}")
		print(f"Shape of labels: {self.labels.shape}")

		# Split the dataset into training and testing data
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)
  
	def preprocess_dataset(self):
	 
		print("Before SMOTE:", Counter(self.y_train))
		print("Shape of X_train:", self.X_train.shape)
		print("Shape of y_train:", self.y_train.shape)
		
		X_train_temp = np.array(self.X_train).reshape(-1, 3)
		y_train_temp = []
		for i in range(self.y_train.shape[0]):
			y_train_temp.extend([self.y_train[i]] * self.LEN_FEATURES)
		y_train_temp = np.array(y_train_temp).reshape(-1, 1)
  
		oversample = SMOTE()
		undersample = RandomUnderSampler(sampling_strategy=1.0)
		steps = [('o', oversample), ('u', undersample)]
		pipeline = Pipeline(steps=steps)
  
		X_train_new, y_train_new = pipeline.fit_resample(X_train_temp, y_train_temp)
  
		x_train_SMOTE = X_train_new.reshape(int(X_train_new.shape[0] / self.LEN_FEATURES), 3, self.LEN_FEATURES)
		y_train_SMOTE = []
		for i in range(int(X_train_new.shape[0] / self.LEN_FEATURES)):
			value_list = list(y_train_new.reshape(int(X_train_new.shape[0] / self.LEN_FEATURES), self.LEN_FEATURES)[i])
			y_train_SMOTE.extend(list(set(value_list)))
			if len(set(value_list)) != 1:
				print('\n\n********* STOP: THERE IS SOMETHING WRONG IN TRAIN ******\n\n')
	
		self.X_train = np.array(x_train_SMOTE)
		self.y_train = np.array(y_train_SMOTE)
  
		print("After SMOTE:", Counter(self.y_train))
		print("New shape of X_train:", self.X_train.shape)
		print("New shape of y_train:", self.y_train.shape)

	def build_model(self):
		
		model = Sequential()

		# Add recurrent layer with 100 units
		model.add(LSTM(100, input_shape=(3, self.LEN_FEATURES)))

		# Add a dropout layer to prevent overfitting
		model.add(Dropout(0.2))

		# Add a dense layer with a single output
		model.add(Dense(1, activation='sigmoid'))

		# Compile the model
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		self.model = model

		return model

	def fit_model(self):
		
		self.model.fit(self.X_train, self.y_train, batch_size=self.BATCH_SIZE, epochs=10, validation_split=0.2)
	
	def evaluate_model(self):
    
		loss, accuracy = self.model.evaluate(self.X_test, self.y_test, batch_size=self.BATCH_SIZE)
		print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
	
	def predict_model(self):
		y_pred1 = self.model.predict(self.X_test)
		y_pred = np.argmax(y_pred1, axis=1)

		# Print f1, precision, and recall scores
		print("Precision:", precision_score(self.y_test, y_pred , average="macro"))
		print("Recall:", recall_score(self.y_test, y_pred , average="macro"))
		print("F1 Score:", f1_score(self.y_test, y_pred , average="macro"))
  
	def make_prediction(self, input_data):
		
		# Scale the input data
		input_data = input_data / input_data.max(axis=2, keepdims=True)

		# Make a prediction
		prediction = self.model.predict(input_data)
  
		return prediction


if __name__ == '__main__':
	census_rnn = CensusRNN()
	census_rnn.load_dataset()
	# census_rnn.preprocess_dataset()
	census_rnn.build_model()
	census_rnn.fit_model()
	# census_rnn.evaluate_model()
	census_rnn.predict_model()
