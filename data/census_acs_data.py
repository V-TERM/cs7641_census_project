"""
File containing functions to collect and preprocess data
collected from American Community Survey (ACS) data.

This website provides:
- a multitude of potential features to use in model
"""

import requests
import json
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
        url = "https://api.census.gov/data/2018/acs/acs5?get="+\
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
