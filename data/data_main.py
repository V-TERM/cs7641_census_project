"""
'Main' script for collecting and preprocessing data for project.
"""

from atlas_data import collect_atlas, preprocess_atlas
from census_acs_data import collect_acs, preprocess_acs
from census_redistrict_data import collect_redistrict, preprocess_redistrict
from mit_data import collect_mit, preprocess_mit
import os.path as osp


class CensusProjectData(object):
    def __init__(self, root_path="./tmp"):
        self.root_path = root_path
        
        # Collect all data.
        self.collect_atlas_data()
        pass

        # Preprocess all data.
        pass

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

if __name__ == '__main__':
    cpd = CensusProjectData()