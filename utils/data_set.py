import torch.utils.data as tordata
class DataSet(tordata.Dataset):
    def __init__(self, label, level, cache) -> None:
        self.label_set = set(self.label)
    

    def __getitem__(self, index) -> T_co:
        return super().__getitem__(index)
    
