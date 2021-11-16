"""
'Main' script for collecting and preprocessing data for project.
"""

from atlas_data import collect_atlas, preprocess_atlas
from census_acs_data import find_common_json, pull_Json, YEARS
from census_acs_data import collect_acs, preprocess_acs
from census_redistrict_data import collect_redistrict, preprocess_redistrict
from data_preprocessing import preprocess_senate_results, preprocess_presidential_results
from mit_data import collect_mit, preprocess_mit
import os.path as osp
import glob


class CensusProjectData(object):
    def __init__(self, root_path="./tmp"):
        self.root_path = root_path
        
        # Collect all data.
        self.collect_atlas_data()
        self.collect_acs_data()
        

        # Preprocess all data.
        self.preprocess_data()
    
    def collect_acs_data(self):
        if not osp.isdir(osp.join(self.root_path, "2009")):
            for year in YEARS:
                pull_Json(year)
            find_common_json(YEARS)

        if not glob.glob(osp.join(self.root_path, "acs_data_*.csv")):
            collect_acs()  # note: may run for 24+ hours
        preprocess_acs()

    def collect_atlas_data(self):
        """Collect data from US Election Atlas, save to .csv."""
        pred_state = osp.join(self.root_path, "pres_state.csv")
        sen_state = osp.join(self.root_path, "sen_state.csv")
        pred_cnty = osp.join(self.root_path, "pres_cnty.csv")
        sen_cnty = osp.join(self.root_path, "sen_cnty.csv")

        if not osp.isfile(pred_state) or not osp.isfile(sen_state) \
            or not osp.isfile(pred_cnty) or not osp.isfile(sen_cnty):
            atlas_data = collect_atlas()
            atlas_data[0].to_csv(pred_state, index=False)
            atlas_data[1].to_csv(sen_state, index=False)
            atlas_data[2].to_csv(pred_cnty, index=False)
            atlas_data[3].to_csv(sen_cnty, index=False)

    def preprocess_data(self):
        pres_county_df, pres_state_df = preprocess_presidential_results()
        sen_county_df, sen_state_df = preprocess_senate_results()

if __name__ == '__main__':
    cpd = CensusProjectData()