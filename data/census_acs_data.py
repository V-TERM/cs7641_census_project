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

YEARS = [str(i) for i in range(2009, 2020)]  # 2009 to 2019


def pull_Json(Year):
    groups_url = F"https://api.census.gov/data/{Year}/acs/acs5/profile/groups.json"
    variables_url = F"https://api.census.gov/data/{Year}/acs/acs5/profile/variables.json"
    tags_url = F"https://api.census.gov/data/{Year}/acs/acs51/tags.json"
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

def inspect_variables_across_years(code_2009, fix_dic={}):
    variables_jsons = {}

    for year in YEARS:
        variables_json = json.load(open(F'./tmp/{year}/variables.json'))['variables']
        variables_jsons[year] = variables_json

    years = [str(i) for i in range(2010, 2020)]  # 2010 to 2019

    # Get description of code_2009 from 2009 to 2019
    for year in YEARS:
        print("Description for", year, ":")

        if year in fix_dic.keys():
            code_year = fix_dic[year]
        else:
            code_year = code_2009

        print(variables_jsons[year][code_year]["label"])
        print()


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

def get_acs_variables_by_year():
    """
    Create list of variables per year using 2009 variables.
    This is due to the fact that the same variable can have different codes
        from year to year.
    """
    
    var_2009 = []
    state = "26"
    cnty = "001"
    YEARS = ["2009"]

    #
    # Variables from DP02 - social characteristics
    #
    dp02 = []
    dp02 += ['0036']  # Number of births
    dp02 += ['0052'] + ['%04d'%(i) + 'P' for i in range(53, 58)]  # School enrollment
    dp02 += ['0058'] + ['%04d'%(i) + 'P' for i in range(59, 68)]  # Educational attainment
    dp02 += ['%04d'%(i) + 'P' for i in (80, 81, 82, 85)]  # Residence 1 year ago
    for i in range(len(dp02)):
        dp02[i] = 'DP02_' + dp02[i] + 'E'
    var_2009 += dp02

    #
    # Variables from DP03 - economic characteristics
    #
    #dp03 = []
    #dp03 += ['0001'] + ['%04d'%(i) + 'P' for i in range(2, 18)]  # Employment status
    #dp03 += ['0026'] + ['%04d'%(i) + 'P' for i in range(27, 33)]  # Occupation
    #dp03 += ['0033'] + ['%04d'%(i) + 'P' for i in range(34, 37)]  # Industry
    #dp03 += ['0047'] + ['%04d'%(i) + 'P' for i in range(48, 52)]  # Class of worker
    #dp03 += ['0052'] + ['%04d'%(i) + 'P' for i in range(53, 96)]  # Income and benefits
    #dp03 += ['0096'] + ['%04d'%(i) + 'P' for i in range(97, 103)]  # Health insurance coverage
    #dp03 += ['%04d'%(i) + 'P' for i in range(103, 122)]  # % families below poverty line
    #for i in range(len(dp03)):
    #    dp03[i] = 'DP03_' + dp03[i] = 'E'
    #var_2009 += dp03

    #
    # Variables from DP04 - housing characteristics
    #
    #dp04 = []
    #dp04 += ['0001'] + ['%04d'%(i) + 'P' for i in range(2, 6)]  # Housing occupancy
    #dp04 += ['0079'] + ['%04d'%(i) + 'P' for i in range(80, 89)]  # Housing value
    #dp04 += ['0108'] + ['%04d'%(i) + 'P' for i in range(108, 124)]  # Selected monthly owner cost as a % of household income
    #dp04 += ['0134'] + ['%04d'%(i) + 'P' for i in range(135, 142)]  # Gross rent as a % of household income
    #for i in range(len(dp04)):
    #    dp04[i] = 'DP04_' + dp04[i] + 'E'
    #var_2009 += dp04

    #
    # Variables from DP05 - demographic characteristics
    #
    #dp05 = []
    #dp05 += ['0001']  # Total population (use to normalize CBP data)
    #dp05 += ['0003']  # Total number of women (use to normalize number of births)
    #for i in range(len(dp05)):
    #    dp05[i] = 'DP05_' + dp05[i] + 'E'
    #var_2009 += dp05

    # TODO; do any manual fixes necessary

    acs_variables_string = ','.join(var_2009)

    # Retrieve data using API
    data = [[] for _ in range(len(var_2009))]

    for year in YEARS:
        url = "https://api.census.gov/data/"+year+"/acs/acs5/profile?get="+\
                acs_variables_string+"&for=county:"+cnty+"&in=state:"+state+\
                "&key="+API_key
        response = requests.get(url)

        try:
            parsed = json.loads(response.text)
            for i in range(len(var_2009)):
                data[i] += [parsed[1][i]]
        except Exception as e:
            print(e)

    # Export to csv: rows = codes, columns = years
    df = pd.DataFrame(data, columns=YEARS, index=var_2009)
    df.to_csv("inspect_variables.csv")

def collect_acs():
    """
    Uses API key to scrape data from Census ACS.
    Returns collected data.
    """

    #acs_variables = get_acs_variables_by_year()
    acs_variables = ['DP03_0{}E'.format(i) for i in range(103, 122)]
    #acs_variables = ['DP03_0{}PE'.format(i) for i in range(119, 137)]
    acs_variables.sort()
    acs_variables_string = ','.join(acs_variables)

    #fips_codes=
    fips_codes = ['26163']

    #years = [str(i) for i in range(2009, 2020)]  # 2009 to 2019
    #YEARS = ["2019"]
    YEARS = ["2009"]

    df_a = []

    for year in YEARS:
        for code in fips_codes:
            state, cnty = code[:2], code[2:]
            print("Pulling data from state", state, "and county", cnty)

            #print(cid)
            url = "https://api.census.gov/data/"+year+"/acs/acs5/profile?get="+\
                    acs_variables_string+"&for=county:"+cnty+"&in=state:"+state+\
                    "&key="+API_key
            response = requests.get(url)

            try:
                parsed = json.loads(response.text)
                data = np.array(parsed)
                df_a += [pd.DataFrame(data=data[1:,0:], columns=data[0,0:])] # [values + 1st row as the column names]
            except:
                print('   Could not retrieve for code %s'%code)

    df = pd.concat(df_a)

    # save and load them, this allows strings become numbers
    df.to_csv('tmp/block_group_tabular.csv', index=False)

def preprocess_acs():
    # TODO
    pass

if __name__ == '__main__':
    #inspect_variables_across_years("DP02_0036E")
    inspect_variables_across_years("DP02_0036E", {"2019": "DP02_0037E"})
    #find_common_json(YEARS)
    #get_acs_variables_by_year()
    #collect_acs()
