"""
File containing functions to collect and preprocess data
collected from American Community Survey (ACS) data.

This website provides:
- a multitude of potential features to use in model
"""
import copy
import json
import os
import os.path as osp
import urllib.request
from collections import defaultdict
import numpy as np
from utils import fill_missing_values_rfr
import pandas as pd
import requests
import glob
import sys

API_key1 = "f937f9f5dd00b893a8dff1bb3cd8936f6ba8e577"
API_key2 = "1d9f93096f2227878a5548c8d104f4ffc6029c5f"
API_key3 = "e23d47b31eb22707ec4a04ba0f9c079259ece862"
API_key4 = "f2f196b80633a6bc2545af4525eaef879ca9581b"

FIPS_TO_STATE = {'53': 'WA', '10': 'DE', '11': 'DC', '55': 'WI', '54': 'WV',
 '15': 'HI', '12': 'FL', '56': 'WY', '46': 'SD', '34': 'NJ', '35': 'NM',
 '48': 'TX', '22': 'LA', '37': 'NC', '38': 'ND', '31': 'NE', '47': 'TN',
 '36': 'NY', '42': 'PA', '02': 'AK', '32': 'NV', '33': 'NH', '51': 'VA',
 '08': 'CO', '06': 'CA', '01': 'AL', '05': 'AR', '50': 'VT', '17': 'IL',
 '13': 'GA', '18': 'IN', '19': 'IA', '25': 'MA', '04': 'AZ', '16': 'ID',
 '09': 'CT', '23': 'ME', '24': 'MD', '40': 'OK', '39': 'OH', '49': 'UT',
 '29': 'MO', '27': 'MN', '26': 'MI', '44': 'RI', '20': 'KS', '30': 'MT',
 '28': 'MS', '45': 'SC', '21': 'KY', '41': 'OR'}

YEARS = [str(i) for i in range(2009, 2020)]  # 2009 to 2019
ELEC_YEARS = ["2009", "2010", "2012", "2014", "2016", "2018", "2019"]


def pull_Json(Year):
    groups_url = F"https://api.census.gov/data/{Year}/acs/acs5/profile/groups.json"
    variables_url = F"https://api.census.gov/data/{Year}/acs/acs5/profile/variables.json"
    tags_url = F"https://api.census.gov/data/{Year}/acs/acs5/tags.json"
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

def inspect_variables_across_years(code_2009, fix_dic={}, print_labels=True, print_csv_line=True):
    if print_labels:
        print()
        print(code_2009)
    
    variables_jsons = {}

    for year in YEARS:
        variables_json = json.load(open(F'./tmp/{year}/variables.json'))['variables']
        variables_jsons[year] = variables_json

    # Get description of code_2009 from 2009 to 2019
    code_years = []
    for year in YEARS:
        if year in fix_dic.keys():
            code_year = fix_dic[year]
        else:
            code_year = code_2009
        code_years += [code_year]

        if print_labels:
            print("Description for", year, ":")
            print(variables_jsons[year][code_year]["label"])
    if print_csv_line:
        print(",".join(code_years))

def get_acs_variables_by_year():
    """
    Create list of variables per year using 2009 variables.
    Write to csv, so data can be inspected for one year.
    """
    
    var_2009 = []
    state = "26"
    cnty = "001"
    years = ["2009"]

    #
    # Variables from DP02 - social characteristics
    #
    # dp02 = []
    # dp02 += ['0036P']  # Number of births
    # dp02 += ['0052P'] + ['%04d'%(i) + 'P' for i in range(53, 58)]  # School enrollment
    # dp02 += ['0058'] + ['%04d'%(i) + 'P' for i in range(59, 68)]  # Educational attainment
    # dp02 += ['%04d'%(i) + 'P' for i in (80, 81, 82, 85)]  # Residence 1 year ago
    # for i in range(len(dp02)):
    #     dp02[i] = 'DP02_' + dp02[i] + 'E'
    # var_2009 += dp02

    #
    # Variables from DP03 - economic characteristics
    #
    #dp03 = []
    #dp03 += ['0001'] + ['%04d'%(i) + 'P' for i in range(2, 8)]  # Employment status
    #dp03 += ['0026'] + ['%04d'%(i) + 'P' for i in range(27, 33)]  # Occupation
    #dp03 += ['0033'] + ['%04d'%(i) + 'P' for i in range(34, 37)]  # Industry
    #dp03 += ['0047'] + ['%04d'%(i) + 'P' for i in range(48, 52)]  # Class of worker
    #dp03 += ['0052'] + ['%04d'%(i) + 'P' for i in range(53, 96)]  # Income and benefits
    #dp03 += ['0096'] + ['%04d'%(i) + 'P' for i in range(97, 103)]  # Health insurance coverage
    #dp03 += ['%04d'%(i) + 'P' for i in range(103, 122)]  # % families below poverty line
    #for i in range(len(dp03)):
    #   dp03[i] = 'DP03_' + dp03[i] + 'E'
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
    dp05 = []
    #dp05 += ['0001']  # Total population (use to normalize CBP data)
    #dp05 += ['0003']  # Total number of women (use to normalize number of births)
    #dp05 += ['%04d'%(i) + 'P' for i in range(4, 17)]
    #dp05 += ['0017']
    #dp05 += ['%04d'%(i) + 'P' for i in range(18, 22)]
    #dp05 += ['%04d'%(i) + 'P' for i in range(32, 65)]
    dp05 += ['%04d'%(i) + 'P' for i in range(65, 81)]
    for i in range(len(dp05)):
        dp05[i] = 'DP05_' + dp05[i] + 'E'
    var_2009 += dp05

    # TODO; do any manual fixes necessary

    acs_variables_string = ','.join(var_2009)

    # Retrieve data using API
    data = [[] for _ in range(len(var_2009))]

    for year in years:
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
    df = pd.DataFrame(data, columns=years, index=var_2009)
    df.to_csv("./tmp/inspect_variables.csv")

def collect_acs(fips_range, api_key, outfile_index):
    """
    Uses API key to scrape data from Census ACS.
    Returns collected data.
    """

    # Get variable list
    var_df = pd.read_csv("profile_variables_by_year.csv")
    num_var = len(var_df)

    var_by_year = {}
    for year in ELEC_YEARS:
        acs_variables_year = var_df[year].to_list()
        var_by_year[year] = acs_variables_year

    # Get fips codes
    with open('./us_county_fips.json', 'r') as f:
        fips_dict = json.load(f)

    main_df = {}
    for i in range(0, num_var, 50):
        main_df[i] = []

    for year in ELEC_YEARS:
    
        df_a = copy.deepcopy(main_df)
        for code, cnty_name in fips_dict.items():
            state, cnty = code[:2], code[2:]

            if int(state) < fips_range[0] or int(state) > fips_range[1]:
                continue

            print("Pulling data from state", state, "and county", cnty, "for year", year)

            df_yc = {}
            for i in range(0, num_var, 50):
                #print(i)
                b = i
                e = min(i + 50, num_var)
                acs_variables_string = ','.join(var_by_year[year][b:e])

                try:
                    url = "https://api.census.gov/data/"+year+"/acs/acs5/profile?get="+\
                        acs_variables_string+"&for=county:"+cnty+"&in=state:"+state+\
                        "&key="+api_key
                    response = requests.get(url)

                    parsed = json.loads(response.text)
                    data = np.array(parsed)

                    if num_var - i <= 50:  # include state + county only once
                        resp_df = pd.DataFrame(data=data[1:,:],
                            columns=pd.RangeIndex(data[0,:].shape[0]))
                        resp_df["state_fips"] = [state for _ in range(len(resp_df))]
                        resp_df["cnty_fips"] = [cnty for _ in range(len(resp_df))]
                        resp_df["state_name"] = [FIPS_TO_STATE[state] for _ in range(len(resp_df))]
                        resp_df["cnty_name"] = [cnty_name.replace(" ", "") for _ in range(len(resp_df))]
                        resp_df["year"] = [year for _ in range(len(resp_df))]
                    else:
                        resp_df = pd.DataFrame(data=data[1:,:-2],
                            columns=pd.RangeIndex(data[0,:-2].shape[0]))

                    df_yc[i] = [resp_df]
                    #print(df_yc[i])
                #except:
                #    print('   Could not retrieve for code %s'%code)
                except Exception as e:
                    print(e)

            if len(df_yc) != len(df_a):
                print("WARNING: for code %s not all calls were successful\n"%code)
                continue

            for k, v in df_yc.items():
                df_a[k] = df_a[k] + v

        for k in df_a.keys():
            df_a[k] = pd.concat(df_a[k], ignore_index=True)
        df = pd.concat([v for _, v in df_a.items()], axis=1)

        df.columns = var_by_year["2009"] + \
            ["state", "county", "state_fips", "cnty_fips", "state_name", "cnty_name", "year"]
        df.to_csv('tmp/acs_data_{}_{}.csv'.format(outfile_index, year), index=False)

#def collect_acs():
#    collect_acs((1,56), API_key1, 0)

def preprocess_acs(in_dir, outfile):
    
    cbp_df = pd.read_csv(osp.join(in_dir, "cbp_variables_by_year.csv"), dtype=str)
    mf_df = pd.read_csv(osp.join(in_dir, "acs_mf_variables_by_year.csv"), dtype=str)

    cbp_df = cbp_df.rename(columns={"YEAR": "year"})
    mf_df = mf_df.rename(columns={"YEAR": "year"})

    mf_df = mf_df[mf_df["year"].isin(["2009", "2010", "2012", "2014", "2016", "2018", "2019"])]

    dfs = []
    df_paths = glob.glob(osp.join(in_dir, "acs_data_*.csv"))
    for path in df_paths:
        dfs += [pd.read_csv(path, dtype=str)]
    df = pd.concat(dfs, axis=0)

    df = pd.merge(cbp_df, df, how='inner', on=['state', 'county', 'year'])

    df_non2009 = pd.merge(mf_df, df, how='inner', on=['state', 'county', 'year'])  # non-2009

    df_2009 = pd.merge(mf_df, df, how='right', on=['state', 'county', 'year'])  # 2009
    df_2009 = df_2009[df_2009["year"] == "2009"]
    print(len(df_2009), len(df_non2009))

    df = pd.concat([df_2009, df_non2009], axis=0)

    print("Number of data points:", len(df))
    print("Number of columns (initially):", len(df.columns))

    non_nans = df.count()

    # Get rid of columns with too many (X)s or -88...8s
    badval_cols = []
    for col in df.columns:
        num_x = len(df[df[col] == '(X)'])
        num_neg8s = len(df[df[col] == '-888888888'])
        num_nans = len(df) - non_nans[col]
        n = num_x + num_neg8s + num_nans
        
        if n >= 0.2 * len(df):
            badval_cols += [col]

    df = df.drop(columns=badval_cols)
    print("Number of columns (after drop):", len(df.columns))

    # Convert (X)s and -88...8s to null values
    df = df.replace("(X)", np.nan)
    df = df.replace("-888888888", np.nan)

    # Use random forest regression to complete matrix
    print("Running random forest regression...")
    df[df.columns[3:-4]] = df[df.columns[3:-4]].apply(pd.to_numeric)
    df[df.columns[3:-4]] = fill_missing_values_rfr(df[df.columns[3:-4]])
    print("Finished random forest regression")

    # Normalize certain columns
    df["DP02_0036E"] /= df["DP05_0003E"]
    df["DP02_0052E"] /= df["DP05_0001E"]
    df["DP02_0058E"] /= df["DP05_0001E"]
    df["DP02_0080E"] /= df["DP05_0001E"]
    df["DP02_0085E"] /= df["DP05_0001E"]
    df["DP03_0001E"] /= df["DP05_0001E"]
    df["DP03_0026E"] /= df["DP05_0001E"]
    df["DP03_0047E"] /= df["DP05_0001E"]
    df["DP04_0001E"] /= df["DP05_0001E"]
    df["DP04_0108E"] /= df["DP05_0001E"]
    df["DP04_0115PE"] /= df["DP05_0001E"]
    df["DP04_0134E"] /= df["DP05_0001E"]
    df["DP05_0003E"] /= df["DP05_0001E"]
    df["MOVEDNET"] /= df["DP05_0001E"]
    df["FROMABROAD"] /= df["DP05_0001E"]
    df["POP1YR"] /= df["DP05_0001E"]
    df["TODIFFSTATE"] /= df["DP05_0001E"]
    df["FROMDIFFSTATE"] /= df["DP05_0001E"]
    df["POP1YRAGO"] /= df["DP05_0001E"]
    df["PAYQTR1"] /= df["DP05_0001E"]
    df["EMP"] /= df["DP05_0001E"]
    df["ESTAB"] /= df["DP05_0001E"]
    df["PAYANN"] /= df["DP05_0001E"]

    # Drop certain columns
    df = df.drop(columns=["DP05_0065E"])

    df.to_csv(outfile, index=False)

if __name__ == '__main__':
    #find_common_json(YEARS)
    
    #which_key = int(sys.argv[1])
    #if which_key == 1:
    #    fips_range, x, y = (0, 20), API_key1, 0
    #elif which_key == 2:
    #    fips_range, x, y = (21, 32), API_key2, 1
    #elif which_key == 3:
    #    fips_range, x, y = (33, 46), API_key1, 2
    #elif which_key == 4:
    #    fips_range, x, y = (47, 56), API_key4, 3
    #print(fips_range, x, y)
    #collect_acs(fips_range, x, y)

    # inspect = False
    # if inspect:
    #     begin, end = 66, 81
    #     group = 5
    #     p_string = "P"
    #     for i in range(begin, end):
    #         code_2009 = "DP0{}_{}{}E".format(group, '%04d'%(i), p_string)
    #         fix_dic = {"2019": "DP0{}_{}{}E".format(group, '%04d'%(i + 5), p_string),
    #         "2018": "DP0{}_{}{}E".format(group, '%04d'%(i + 5), p_string),
    #         "2017": "DP0{}_{}{}E".format(group, '%04d'%(i + 5), p_string),}
    #         inspect_variables_across_years(code_2009, fix_dic, True, False)
    #     c = begin - 1
    #     inspect_variables_across_years("DP0{}_{}E".format(group, '%04d'%(c)),
    #         fix_dic = {"2019": "DP0{}_{}E".format(group, '%04d'%(c + 5)),
    #         "2018": "DP0{}_{}E".format(group, '%04d'%(c + 5)),
    #         "2017": "DP0{}_{}E".format(group, '%04d'%(c + 5)),})
    #     for i in range(begin, end): 
    #         code_2009 = "DP0{}_{}{}E".format(group, '%04d'%(i), p_string)
    #         fix_dic = {"2019": "DP0{}_{}{}E".format(group, '%04d'%(i + 5), p_string),
    #         "2018": "DP0{}_{}{}E".format(group, '%04d'%(i + 5), p_string),
    #         "2017": "DP0{}_{}{}E".format(group, '%04d'%(i + 5), p_string),}
    #         inspect_variables_across_years(code_2009, fix_dic, False, True)
    # else:
    #     get_acs_variables_by_year()
    preprocess_acs("./tmp", "./tmp/census_data.csv")
