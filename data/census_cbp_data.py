"""
File containing functions to collect and preprocess data
collected from Community Business Patterns data.

This website provides:
- a multitude of potential features to use in model
"""
import json
import os
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

API_key = "f937f9f5dd00b893a8dff1bb3cd8936f6ba8e577"

YEARS = [str(i) for i in range(2009, 2020)]  # 2009 to 2019

def pull_data_from_json(filename='cbp.json'):
	if not os.path.exists(filename):
		return

	with open(filename) as f:
		content = json.load(f)
		raw_data = content['dataset']
	
	for raw_data_y in raw_data:
		y = raw_data_y['c_vintage']
		groups_url = raw_data_y['c_groupsLink']
		variables_url = raw_data_y['c_variablesLink']
		tags_url = raw_data_y['c_tagsLink']


		if not os.path.isdir(F'./tmp/{y}'):
			os.makedirs(F'./tmp/{y}')
		try:
			urllib.request.urlretrieve(groups_url, F'./tmp/{y}/groups.json')
		except:
			print("Groups", y)
		try:
			urllib.request.urlretrieve(variables_url, F'./tmp/{y}/variables.json')
		except:
			print("Variables", y)
		try:
			urllib.request.urlretrieve(tags_url, F'./tmp/{y}/tags.json')
		except:
			print("Tags", y)

def get_common_variables():
	"""
	Get common variables from ACS and CBP.
	"""
	common_variables = []

	for y in YEARS:
		path = os.path.join('./tmp/', y)
		with open(os.path.join(path, 'variables.json')) as f:
			content = json.load(f)
			variables = content['variables']
		
			if common_variables == []:
				common_variables = list(variables.keys())
			else:
				for v in common_variables:
					if v not in variables.keys():
						del common_variables[common_variables.index(v)]
	

	remove_variables = ['for', 'in', 'EMP_N', 'PAYQTR1_N', 'CSA', 'LFO', 'PAYANN_N', 'GEO_ID', 'COUNTY', 'EMPSZES']
	for v in remove_variables:
		del common_variables[common_variables.index(v)]
	
	print(common_variables)
	return common_variables

def get_cbp_variables(common_variables):
	"""
	Create list of variables per year using 2009 variables.
	This is due to the fact that the same variable can have different codes
		from year to year.
	"""

	for y in YEARS:

		print("Getting variables for year", y)
		# if os.path.exists(F'./tmp/{y}/variables.json'):
		# 	continue

		cbp_variables_string = ','.join(common_variables)
		url = "https://api.census.gov/data/" + y + "/cbp?get=" + cbp_variables_string + "&for=county:*&in=state:*&key=" + API_key
		response = requests.get(url)

		try:
			parsed = json.loads(response.text)
			data = np.array(parsed)
			df = pd.DataFrame(data=data[1:, 0:], columns=data[0, 0:])  # [values + 1st row as the column names]
			df = df.sort_values(by=['state', 'county'])
			df.to_csv(F'./tmp/{y}/cbp_variables.csv', index=False)
		except Exception as e:
			print(response.text)
			print(e)
			print('Could not retrieve for year %s' % y)
			exit()

def collapse_csv():
	dfs = []
	for year in range(2009, 2020):
		filepath = "./tmp/" + str(year) + "/cbp_variables.csv"
		dfs.append(pd.read_csv(filepath))
	df = pd.concat(dfs)
	df = df.sort_values(by=['YEAR', 'state', 'county'])

	# combine rows of same county and state by adding the values
	df = df.groupby(['YEAR', 'state', 'county']).sum().reset_index()
	df.to_csv('./tmp/cbp_variables_by_year.csv', index=False)

if __name__ == '__main__':
	# pull_data_from_json()
	# common_variables = get_common_variables()
	# get_cbp_variables(common_variables)
	# collapse_csv()
	pass
