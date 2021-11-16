import os
import numpy as np
import pandas as pd
import json
join = os.path.join



class Presidential_Results(object):
    def __init__(self, root_dir='./tmp'):
        self._county_block = pd.read_csv(join(root_dir, 'pres_cnty.csv'))
        self._state_block = pd.read_csv(join(root_dir, 'pres_state.csv'))
        self._census_data = pd.read_csv(join(root_dir, 'use_later.csv'))

    @property
    def counties(self):
        return self._county_block['county'].to_list()

    @property
    def states(self):
        return self._state_block['state'].to_list()

    def get_county(self, cnty):
        election_results = self._county_block[cnty == self._county_block['county']][['cand1_percent', 'cand2_percent', 'cand3_percent', 'year']]        
        census_data = self._census_data[cnty == self._census_data['cnty_name']]
    
        census_year = census_data['year'].to_numpy()
        years = election_results['year'].to_numpy()

        election_year_msk = election_results['year'].isin(census_year)
        election_results = election_results.drop(columns=['year']).to_numpy()

        census_data = census_data.loc[census_data['year'].isin(years)]
        census_data = census_data.drop(columns=['state', 'county', 'state_fips', 'cnty_fips', 'state_name', 'cnty_name', 'year'])
        census_data = census_data.to_numpy()
        
        winners = np.argmax(election_results, axis=1)
        label = np.zeros((election_results.shape[0], 1))

        for i in range(1, election_results.shape[0]):
            last_winner = winners[i-1]
            current_winner = winners[i]
            label[i,:] = election_results[i, last_winner] - election_results[i-1, current_winner] 
            if last_winner == current_winner:
                label[i, :] = np.abs(label[i, :])            
        label = label[election_year_msk]
        return census_data, label

    def get_state(self, state):
        election_results = self._state_block[state == self._state_block['state']][['cand1_percent', 'cand2_percent', 'cand3_percent', 'year']]        
        census_data = self._census_data[state == self._census_data['state_name']]
        census_year = census_data['year'].to_numpy()
        years = election_results['year'].to_numpy()

        election_year_msk = election_results['year'].isin(census_year)
        election_results = election_results.drop(columns=['year']).to_numpy()

        census_data = census_data.loc[census_data['year'].isin(years)]
        census_data = census_data.drop(columns=['state', 'county', 'state_fips', 'cnty_fips', 'state_name', 'cnty_name', 'year'])
        census_data = census_data.to_numpy()
        
        winners = np.argmax(election_results, axis=1)
        label = np.zeros((election_results.shape[0], 1))


        

class Senate_Result(object):
    def __init__(self, root_dir='./tmp'):
        self._county_block = pd.read_csv(join(root_dir, 'sen_cnty.csv'))
        self._state_block = pd.read_csv(join(root_dir, 'sen_state.csv'))
        self._census_data = pd.read_csv(join(root_dir, 'use_later.csv'))
        ['Democratic', 'Republican', 'Other']

    @property
    def counties(self):
        return self._county_block['county'].to_list()

    @property
    def states(self):
        return self._state_block['state'].to_list()

    def get_county(self, cnty):
        election_results = self._county_block[cnty == self._county_block['county']][['cand1_percent', 'cand2_percent', 'cand3_percent', 'year']]        
        census_data = self._census_data[cnty == self._census_data['cnty_name']]
    
        census_year = census_data['year'].to_numpy()
        years = election_results['year'].to_numpy()

        election_year_msk = election_results['year'].isin(census_year)
        election_results = election_results.drop(columns=['year']).to_numpy()

        census_data = census_data.loc[census_data['year'].isin(years)]
        census_data = census_data.drop(columns=['state', 'county', 'state_fips', 'cnty_fips', 'state_name', 'cnty_name', 'year'])
        census_data = census_data.to_numpy()
        
        winners = np.argmax(election_results, axis=1)
        label = np.zeros((election_results.shape[0], 1))

        for i in range(1, election_results.shape[0]):
            last_winner = winners[i-1]
            current_winner = winners[i]
            label[i,:] = election_results[i, last_winner] - election_results[i-1, current_winner] 
            if last_winner == current_winner:
                label[i, :] = np.abs(label[i, :])            
        label = label[election_year_msk]
        return census_data, label

    def get_state(self, state):
        census_data = self._census_data[state == self._census_data['state_name']]

if __name__ == "__main__":
    pres_dataloader = Presidential_Results()
    pres_dataloader.get_county("Alcona")
    pres_dataloader.get_state('MI')
    sen_dataloader = Senate_Result()
    sen_dataloader.get_county("Alcona")
    sen_dataloader.get_state('MI')    