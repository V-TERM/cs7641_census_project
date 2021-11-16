import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
join = os.path.join

def calc_pct_change(x):

    for col in x:
        if col not in ['year', 'state_name', 'cnty_name']:
            x[col] = x[col].pct_change()
    print(x.head())
    return x

class Presidential_Results(object):
    def __init__(self, root_dir='./tmp'):
        self._county_block = pd.read_csv(join(root_dir, 'pres_cnty.csv'))
        self._state_block = pd.read_csv(join(root_dir, 'pres_state.csv'))
        self._census_data = pd.read_csv(join(root_dir, 'census_data.csv'))
        
    @property
    def counties(self):
        states = self.states
        county_dict = dict([(s, []) for s in states])

        census_set = self._census_data[['cnty_name', 'state_name']].applymap(lambda x: x.strip() if isinstance(x, str) else x)
        census_set = census_set[census_set['state_name'].isin(states)]
        census_set = census_set.rename(columns={'cnty_name': 'county', 'state_name': 'state'}).to_numpy()
        for county, state in census_set:
            if county not in county_dict[state]:
                county_dict[state].append(county)
        
        # county_set = self._county_block[['county', 'state']].applymap(lambda x: x.strip() if isinstance(x, str) else x)
        # mask = census_set.isin(county_set)
        # combined = census_set[mask]

        county_tuples = []
        for k, v in county_dict.items():
            for _v in v:
                county_tuples += [(_v, k)]

        return county_tuples

    @property
    def states(self):
        census_set = set(pd.unique(self._census_data['state_name']))
        state_set = set(pd.unique(self._state_block['state']))
        combined_set = census_set.intersection(state_set)
        return list(combined_set)

    def get_county(self, cnty, state):
        election_results = self._county_block[cnty == self._county_block['county']]
        election_results = election_results[state == election_results['state']][['cand1_percent', 'cand2_percent', 'cand3_percent', 'year']]        
        election_results = election_results.groupby(pd.Grouper(key='year'), as_index=False).mean()

        census_data = self._census_data[cnty == self._census_data['cnty_name']]
        census_data = census_data[state == census_data['state_name']]
    
        census_year = census_data['year'].to_numpy()
        years = election_results['year'].to_numpy()

        election_year_msk = election_results['year'].isin(census_year)
        election_results = election_results.drop(columns=['year']).to_numpy()

        census_data_mask = census_data['year'].isin(years)
        # census_data = census_data.loc[census_data_mask]
        census_data = census_data.drop(columns=['state', 'county', 'state_fips', 'cnty_fips'])
        census_data = census_data.groupby(pd.Grouper(key='cnty_name'), as_index=False).apply(calc_pct_change)
        
        
        winners = np.argmax(election_results, axis=1)
        label = np.zeros((election_results.shape[0], 1))

        for i in range(1, election_results.shape[0]):
            last_winner = winners[i-1]
            current_winner = winners[i]
            label[i,:] = election_results[i, last_winner] - election_results[i-1, current_winner] 
            
            if last_winner == current_winner:
                # label[i, :] = np.abs(label[i, :])        
                label[i, :] = 0
            else:
                label[i, : ] = 1


        label = label[election_year_msk]
        census_data = census_data[census_data_mask]
        census_data['label'] = label        
        return census_data

    def get_state(self, state):
        election_results = self._state_block[state == self._state_block['state']][['cand1_percent', 'cand2_percent', 'cand3_percent', 'year']]        
        census_data = self._census_data[state == self._census_data['state_name']]
        census_year = np.unique(census_data['year'].to_numpy())
        years = np.unique(election_results['year'].to_numpy())
        election_results = election_results.groupby(pd.Grouper(key='year'), as_index=False).mean()
        election_year_msk = election_results['year'].isin(census_year)
        election_results = election_results.drop(columns=['year']).to_numpy()

        census_data = census_data.groupby(pd.Grouper(key='year'), as_index=False).mean()
        census_data = census_data.loc[census_data['year'].isin(years)]
        census_data = census_data.drop(columns=['state', 'county', 'state_fips', 'cnty_fips'])
        
        winners = np.argmax(election_results, axis=1)
        label = np.zeros((election_results.shape[0], 1))
        for i in range(1, election_results.shape[0]):
            last_winner = winners[i-1]
            current_winner = winners[i]
            label[i,:] = election_results[i, last_winner] - election_results[i-1, current_winner] 
            if last_winner == current_winner:
                label[i, :] = np.abs(label[i, :])   

        label = label[election_year_msk]        
        census_data['label'] = label
        census_data['state'] = state
        return census_data

class Senate_Result(Presidential_Results):
    def __init__(self, root_dir='./tmp'):
        self._county_block = pd.read_csv(join(root_dir, 'sen_cnty.csv'))
        self._state_block = pd.read_csv(join(root_dir, 'sen_state.csv'))
        self._census_data = pd.read_csv(join(root_dir, 'census_data.csv'))


def preprocess_presidential_results():
    _dataloader = Presidential_Results()
    county_df = None
    state_df = None
    for county, state in tqdm(_dataloader.counties, desc='county presidential preprocessing'):
        _df = _dataloader.get_county(county.strip(), state.strip())
        if county_df is None:
            county_df = _df
        else:
            county_df = pd.concat([county_df, _df])

    county_df.to_csv("./county_pres.csv")
    for state in tqdm(_dataloader.states, desc='state presidential preprocessing'):
        _df = _dataloader.get_state(state.strip())
        if state_df is None:
            state_df = _df
        else:
            state_df = pd.concat([state_df, _df])
    state_df.to_csv("./state_pres.csv")
    return county_df, state_df


def preprocess_senate_results():
    _dataloader = Senate_Result()
    county_df = None
    state_df = None
    for county, state in tqdm(_dataloader.counties, desc='county senate preprocessing'):
        _df = _dataloader.get_county(county.strip(), state.strip())
        if county_df is None:
            county_df = _df
        else:
            county_df = pd.concat([county_df, _df])
    county_df.to_csv("./county_sen.csv")

    for state in tqdm(_dataloader.states, desc='state senate preprocessing'):
        _df = _dataloader.get_state(state.strip())
        if state_df is None:
            state_df = _df
        else:
            state_df = pd.concat([state_df, _df])
    state_df.to_csv("./state_sen.csv")
    return county_df, state_df

if __name__ == "__main__":
    df1, df2 = preprocess_presidential_results()
    df1, df2 = preprocess_senate_results()
    