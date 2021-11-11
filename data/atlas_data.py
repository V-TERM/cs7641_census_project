"""
File containing functions to collect and preprocess election data
collected from US Election Atlas (https://uselectionatlas.org/RESULTS/).

This website provides:
- State and county level data for:
    - presidential elections from 2000 to 2020
    - senatorial elections from 2000 to 2020
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd

# NOTE: need to change this value depending on wher chromedriver is installed
DRIVER_PATH = "/usr/local/bin/chromedriver"

# for extacting county data
PRES_ELEC_YRS = ['2000', '2004', '2008', '2012', '2016', '2020']
SEN_ELEC_YRS = ['2000', '2002', '2004', '2006', '2008', '2010', '2012',
                '2014', '2016', '2018', '2020']

FIPS_CODES = {
    'WA': '53', 'DE': '10', 'DC': '11', 'WI': '55', 'WV': '54', 'HI': '15',
    'FL': '12', 'WY': '56', 'SD': '46', 'NJ': '34', 'NM': '35', 'TX': '48',
    'LA': '22', 'NC': '37', 'ND': '38', 'NE': '31', 'TN': '47', 'NY': '36',
    'PA': '42', 'AK': '02', 'NV': '32', 'NH': '33', 'VA': '51', 'CO': '08',
    'CA': '06', 'AL': '01', 'AR': '05', 'VT': '50', 'IL': '17', 'GA': '13',
    'IN': '18', 'IA': '19', 'MA': '25', 'AZ': '04', 'ID': '16', 'CT': '09',
    'ME': '23', 'MD': '24', 'OK': '40', 'OH': '39', 'UT': '49', 'MO': '29',
    'MN': '27', 'MI': '26', 'RI': '44', 'KS': '20', 'MT': '30', 'MS': '28',
    'SC': '45', 'KY': '21', 'OR': '41'
}

COUNTY_COLS = ['year', 'state', 'county', 'cand1_name', 'cand1_percent',
               'cand2_name', 'cand2_percent', 'cand3_name', 'cand3_percent',
               'cand4_name', 'cand4_percent']
STATE_COLS = ['state', 'year', 'cand1_party', 'cand1_count', 'cand1_percent',
              'cand2_party', 'cand2_count', 'cand2_percent',
              'cand3_party', 'cand3_count', 'cand3_percent',
              'cand4_party', 'cand4_count', 'cand4_percent']


# Find longest common substring in list of strings
# Source: https://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings
def long_substr(d):
    substr = ''
    if len(d) > 1 and len(d[0]) > 0:
        for i in range(len(d[0])):
            for j in range(len(d[0])-i+1):
                if j > len(substr) and is_substr(d[0][i:i+j], d):
                    substr = d[0][i:i+j]
    return substr
def is_substr(find, d):
    if len(d) < 1 and len(find) < 1:
        return False
    for i in range(len(d)):
        if find not in d[i]:
            return False
    return True


def visit_atlas_page_county(driver, year, state, office):
    """
    Collect county-level data given a year, state, and office code.
    Returns list of lists representing data collected.
    """
    
    data = []

    # Construct URL
    url = "https://uselectionatlas.org/RESULTS/datagraph.php?"
    url += "year={}&".format(year)
    url += "fips={}&".format(FIPS_CODES[state])
    url += "f=1&off={}&".format(office)
    url += "elect=0&class=3"
    print("Visiting", url)

    # Scrape data from webpage
    driver.get(url)
    tables = driver.find_elements(By.TAG_NAME, 'table')

    ts = []
    for table in tables:
        t = table.text.split('\n')

        if not t:  # get rid of rows with no values (TODO)
            continue

        for i in range(len(t)):
            t[i] = t[i].strip('\%')
        for i in range(1, len(t)):
            t[i] = t[i][2:]  # strip leading whitespace

        ts += [t]

    # infer first row candidate name
    d = [t[0].replace(' ', '') for t in ts]
    d2 = [x.replace('.', '') for x in d]
    d2 = list(filter(None, d2))
    frc_name = long_substr(d2)  # note: w/o whitespace
    #print(d2)
    #print(frc_name)

    for j, t in enumerate(ts):
        #print(t)
        t = list(filter(None, t))
        if not t:
            continue 

        table_data = [year, state]

        # Add county, first cand, result
        county = d2[j][:-3].replace(frc_name, '')
        frc_res = t[0][-4:]
        table_data += [county, frc_name, frc_res]

        # Add remaining cand. + result
        for i in range(1, len(t)):
            cand_res = t[i].split(' ')[-1]
            cand_name = t[i].strip(' ' + cand_res)
            table_data += [cand_name, cand_res]
        
        if len(table_data) < 11:
            table_data += [''] * (11 - len(table_data))

        assert len(table_data) == 11
        #print(table_data)
        data.append(table_data)

    return data


def visit_atlas_page_state(driver, year, state, office):
    """
    Collect state-level data given a year, state, and office code.
    Returns list of lists representing data collected.
    """
    
    data = []

    # Construct URL
    url = "https://uselectionatlas.org/RESULTS/comparegraphs.php?"
    url += "year={}&".format(year)
    url += "fips={}&".format(FIPS_CODES[state])
    url += "f=1&off={}&".format(office)
    url += "elect=0"
    print("Visiting", url)

    # Scrape data from webpage
    driver.get(url)
    tables = driver.find_elements(By.TAG_NAME, 'table')

    for table in tables:
        table_data = table.text.replace(' (S)', '') # 'S' indicates special election - remove it
        table_data = table_data.split(' ')
        for i in range(len(table_data)):
            table_data[i] = table_data[i].strip('\n')
            table_data[i] = table_data[i].strip('\%')
        table_data = list(filter(None, table_data))

        if not table_data:  # get rid of rows with no values
            continue

        if len(table_data) < 13:  # expect year + 4 cand. results (count + %)
            table_data += [''] * (13 - len(table_data))

        table_data = [state] + table_data
        assert len(table_data) == 14

        #print(table_data)
        data.append(table_data)

    return data


def collect_atlas():
    """
    Uses Selenium browser to scrape data from website.
    Returns collected data.
    """

    pres_data_state, sen_data_state = [], []
    pres_data_cnty, sen_data_cnty = [], []

    # Initialize driver, settings
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(DRIVER_PATH, options=chrome_options)

    # Iterate and scrape all state-level data
    for state in FIPS_CODES.keys():

        # Get presidential election data (code: 0)
        for year in PRES_ELEC_YRS:
            pres_data_state += visit_atlas_page_state(driver, year, state, '0')
            pres_data_cnty += visit_atlas_page_county(driver, year, state, '0')
        # Get senatorial election data (code: 3)
        for year in SEN_ELEC_YRS:
            sen_data_state += visit_atlas_page_state(driver, year, state, '3')
            sen_data_cnty += visit_atlas_page_county(driver, year, state, '3')

    pres_data_state = pd.DataFrame(pres_data_state, columns=STATE_COLS)
    sen_data_state = pd.DataFrame(sen_data_state, columns=STATE_COLS)
    pres_data_cnty = pd.DataFrame(pres_data_cnty, columns=COUNTY_COLS)
    sen_data_cnty = pd.DataFrame(sen_data_cnty, columns=COUNTY_COLS)

    return pres_data_state, sen_data_state, pres_data_cnty, sen_data_cnty


def preprocess_atlas():
    # TODO
    pass
