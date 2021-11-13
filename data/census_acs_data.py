"""
File containing functions to collect and preprocess data
collected from American Community Survey (ACS) data.

This website provides:
- a multitude of potential features to use in model
"""
from collections import defaultdict
import os
import requests
import json
import urllib.request
import pandas as pd
import numpy as np

API_key = "f937f9f5dd00b893a8dff1bb3cd8936f6ba8e577"


# TODO: this is code that i mostly copied from a previous project where i worked with census ACS data
# TODO: this provides a template for retrieving ACS variables relevant to our project
def collect_acs():
    """
    Uses API key to scrape data from Census ACS.
    Returns collected data.
    """

    # TODO: make these variables relevant to our project
    acs_variables = ['GEO_ID', 'B01003_001E',
                    'B17010_002E', 'B17010_001E',
                    'B25003_003E', 'B25003_001E',
                    'B25064_001E', 'B19013_001E',
                    'B25077_001E', 'B25071_001E',
                    'B02001_002E', 'B02001_003E',
                    'B02001_004E', 'B02001_005E',
                    'B02001_006E', 'B02001_007E',
                    'B02001_008E']

    ## keep these columns
    acs_dic = {'GEO_ID': 'GEOID10',
                'B01003_001E':'population',
                'B02001_002E':'pop_white',
            'B02001_003E':'pop_black_af_am',
            'B02001_004E':'pop_am_indian_ak_native',
            'B02001_005E':'pop_asian',
            'B02001_006E':'pop_nat_hawaiian_pac_islander',
            'B02001_007E':'pop_some_other_race',
            'B02001_008E':'pop_two_or_more_races',
            'B25064_001E':'median_gross_rent',
            'B19013_001E':'median_household_income',
            'B25077_001E':'median_property_value',
            'B25071_001E':'rent_burden'}

    ## drop these columns (they only will be used to derive other quantities)
    acs_drop = ['B17010_002E', 'B17010_001E',
                'B25003_003E', 'B25003_001E']

    acs_variables_string = ','.join(acs_variables)

    county_ids = ['%03d'%(163) ]

    df_a = []

    for cid in  county_ids:
        #print(cid)
        url = "https://api.census.gov/data/2018/acs/acs1?get="+\
                acs_variables_string+"&for=block%20group:*&in=state:26%20county:"+cid+"&key="+API_key
        response = requests.get(url)

        try:
            parsed = json.loads(response.text)
            data = np.array(parsed)
            df_a += [pd.DataFrame(data=data[1:,0:], columns=data[0,0:])] # [values + 1st row as the column names]
        except:
            print('   Could not retrieve county %s'%cid)

    df = pd.concat(df_a)
    del data, parsed, response

    # save and load them, this allows strings become numbers
    df.to_csv('tmp/block_group_tabular.csv', index=False)

def preprocess_acs():
    # TODO
    pass



years = [i for i in range(2007, 2020)]
print(years)
def pull_Json(Year):
    groups_url = F"https://api.census.gov/data/{Year}/acs/acs1/groups.json"
    variables_url = F"https://api.census.gov/data/{Year}/acs/acs1/variables.json"
    tags_url = F"https://api.census.gov/data/{Year}/acs/acs1/tags.json"
    if not os.path.isdir(F'./tmp/{Year}'):
        os.makedirs(F'./tmp/{Year}')
    try:
        urllib.request.urlretrieve(groups_url, F'./tmp/{Year}/groups.json')
    except:
        print("Groups", Year)
    try:
        urllib.request.urlretrieve(variables_url, F'./tmp/{Year}/variables.json')
    except:
        print("Variables", Year)
    try:
        urllib.request.urlretrieve(tags_url, F'./tmp/{Year}/tags.json')
    except:
        print("Tags", Year)

# [pull_json(year) for year in years]
def find_common_json(years): 
    groups = defaultdict(int)
    variables = defaultdict(int)   

    for Year in years:
        groups_json = json.load(open(F'./tmp/{Year}/groups.json')) 
        for elem in groups_json['groups']:
            groups[elem['name']] += 1
    
    ret_groups = [k for k, v in groups.items() if v == len(years)]
    print("groups reduced from:", len(groups), "to:", len(ret_groups))


    for Year in years:
        variables_json = json.load(open(F'./tmp/{Year}/variables.json'))['variables']
        for var in variables_json.keys():
            if var not in ['for', 'in', 'ucgid']:
                variables[var] += 1

    ret_variables = [k for k, v in variables.items() if v == len(years)]
    print("variables reduced from:", len(variables), "to:", len(ret_variables))

    groups_json = json.load(open(F'./tmp/2019/groups.json'))['groups']
    variables_json = json.load(open(F'./tmp/2019/variables.json'))['variables']
    
    groups_dict = {}
    variables_dict = {}
    for g in groups_json:
        if g['name'] in ret_groups:
            groups_dict[g['name']] = g
    for v in ret_variables:
        variables_dict[v] = variables_json[v]

    json.dump(groups_dict, open('./tmp/groups.json', 'w'))
    json.dump(variables_dict, open('./tmp/variables.json', 'w'))


    


find_common_json(years)