from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class CensusPCA():
	def __init__(self):
		self.data = None  # quantitative data (to PCA)
		self.data_norm = None  # PCA data, normalized
		self.other_cols = None  # non-quantitative data
		self.X_min = None  # PCA result

	def import_data(self, file_name):
		# Import data from csv file
		df = pd.read_csv(file_name)

		self.data = df[df.columns[3:-4]]

		# subtract the mean
		self.data_norm = StandardScaler().fit_transform(self.data)

		self.other_cols = df[list(df.columns[:3]) + \
							list(df.columns[-4:])]

	def run_pca(self, retained_variance=0.99):
		pca = PCA()
		X = pca.fit_transform(self.data_norm)
		evr = pca.explained_variance_ratio_
		evr_cumul = np.cumsum(evr)
		for i, var in enumerate(evr_cumul):
			if var >= retained_variance:
				k = i + 1
				break
		print("Dimensions retained:", k)
		self.X_min = X[:, :k]

	def write_to_file(self, outfile="./data/tmp/census_data_pca.csv"):
		df = pd.DataFrame(self.X_min,
						  columns=pd.RangeIndex(self.X_min.shape[1]))
		df = pd.concat([df, self.other_cols], axis=1)
		df.to_csv(outfile, index=False)

if __name__ == '__main__':
	pca = CensusPCA()
	pca.import_data("./data/tmp/census_data.csv")
	pca.run_pca()
	pca.write_to_file()
