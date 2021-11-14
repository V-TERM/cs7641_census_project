import os
import pandas as pd
join = os.path.join


class Presidential_Results(object):
    def __init__(self, root_dir='./tmp'):
        _county_block = pd.read_csv(join(root_dir, 'pres_cnty.csv'))
        _state_block = pd.read_csv(join(root_dir, 'pres_state.csv'))
        canidates = _county_block.loc[:, _county_block.columns.str.endswith("_name")].unique()
        states = list(pd.unique(_state_block['state']))
        print(canidates.head())

class Senate_Result(object):
    def __init__(self, root_dir='./tmp'):
        _county_block = pd.read_csv(join(root_dir, 'sen_cnty.csv'))
        _state_block = pd.read_csv(join(root_dir, 'sen_state.csv'))
        ['Democratic', 'Republican', 'Other']
        canidates = list(pd.unique(_county_block.filter(regex="cand*_name")))
        states = list(pd.unique(_state_block['state']))
        print(_county_block.head())        


if __name__ == "__main__":
    pres_dataloader = Presidential_Results()
    sen_dataloader = Senate_Result()
    