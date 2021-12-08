import numpy as np
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 128
LEN_FEATURES = 100

class CensusRNN:
	def __init__(self, batch_size=128, no_features=100) -> None:
		self.BATCH_SIZE = batch_size
		self.LEN_FEATURES = no_features

		self.model = None

	def load_dataset(self, filename):
		self.data = np.random.random((1000, 3, 100))
		self.labels = np.random.randint(2, size=(1000, 1))
  
		# scale data between 0 and 1 without standardscaler
		self.data = self.data / self.data.max(axis=2, keepdims=True)

		# Split the dataset into training and testing data
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

	def build_model(self):
		
		model = Sequential()
  
		# Add a masking layer to the model
		# model.add(Masking(mask_value=0., input_shape=(3, LEN_FEATURES)))

		# Add recurrent layer with 100 units
		model.add(LSTM(100, input_shape=(3, LEN_FEATURES), return_sequences=True))

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


if __name__ == '__main__':
	census_rnn = CensusRNN()
	census_rnn.load_dataset('data/census_rnn.csv')
	census_rnn.build_model()
	census_rnn.fit_model()
	census_rnn.evaluate_model()
