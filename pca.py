import pandas as pd


class PCA():

	def __init__(self) -> None:
		self.data = None

	def import_data(self, file_name: str) -> pd.DataFrame:
		# Import data from csv file
		df = pd.read_csv(file_name)
		return df

	def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
		# Remove unnecessary columns
		df = df.drop(columns=['Unnamed: 0'])
		return df
		