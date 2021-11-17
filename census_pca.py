from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CensusPCA():
	def __init__(self):
		self.data = None  # quantitative data (to PCA)
		self.data_norm = None  # PCA data, normalized
		self.other_cols = None  # non-quantitative data
		self.X_min = None  # PCA result
		self.evr_cumul = None
		self.k = None

	def import_data(self, file_name):
		# Import data from csv file
		df = pd.read_csv(file_name)
		df = df.dropna(axis=0)

		if "state" in file_name:
			begin, end = 2, -2
		elif "county" in file_name:
			begin, end = 2, -3

		self.data = df[df.columns[begin:end]]

		# subtract the mean
		self.data_norm = StandardScaler().fit_transform(self.data)

		self.other_cols = df[list(df.columns[:begin]) + \
							list(df.columns[end:])]

	def run_pca(self, retained_variance=0.99):
		pca = PCA()
		X = pca.fit_transform(self.data_norm)
		evr = pca.explained_variance_ratio_
		self.evr_cumul = np.cumsum(evr)
		for i, var in enumerate(self.evr_cumul):
			if var >= retained_variance:
				k = i + 1
				break
		print("Dimensions retained:", k)
		self.k = k
		self.X_min = X[:, :k]

	def write_to_file(self, outfile="./data/tmp/census_data_pca.csv"):
		df = pd.DataFrame(self.X_min,
						  columns=pd.RangeIndex(self.X_min.shape[1]))
		df = pd.concat([df, self.other_cols], axis=1)
		df = df.drop(columns=['Unnamed: 0'], errors='ignore')
		df.to_csv(outfile, index=False)

	def visualize_to_file(self, outfile):
		k = np.arange(1, self.evr_cumul.shape[0] + 1)
		plt.plot(k, self.evr_cumul * 100)
		plt.xlabel("k")
		plt.ylabel("% variance retained")
		plt.title("PCA results: state presidential data")
		plt.plot(self.k, self.evr_cumul[self.k + 1] * 100,'ro')
		plt.text(70,25,"k = {} retains 99% variance".format(self.k), fontsize=12)
		plt.savefig(outfile)

if __name__ == '__main__':
	pca = CensusPCA()
	pca.import_data("./data/state_pres.csv")
	pca.run_pca()
	pca.write_to_file("./data/state_pres_pca.csv")
	pca.visualize_to_file("./data/state_pres_pca.png")
