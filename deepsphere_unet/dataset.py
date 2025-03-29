from torch.utils.data import Dataset
import torch

import healpy as hp

def reorder_map(map, r2n=False, n2r=False):
    unit = map.unit
    data = map.value

    if r2n:
        data = hp.reorder(data, r2n=True)
    elif n2r:
        data = hp.reorder(data, n2r=True)
    else:
        raise ValueError("Either r2n or n2r must be set to True")
    return data * unit
class TrainCMBMapDataset(Dataset):
    def __init__(self, 
                 n_sims,
                 freqs,
                 map_fields: str,
                 label_path_template,
                 label_handler,
                 feature_path_template,
                 feature_handler,
                 ):
        # TODO: Adopt similar method as in parallel operations to allow 
        #       this to use num_workers and transforms
        self.n_sims = n_sims
        self.freqs = freqs
        self.label_path_template = label_path_template
        self.label_handler = label_handler
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields:int = len(map_fields)

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        features = _get_features_idx(freqs=self.freqs,
                                path_template=self.feature_path_template,
                                handler=self.feature_handler,
                                n_map_fields=self.n_map_fields,
                                sim_idx=sim_idx)

        label = _get_label_idx(path_template=self.label_path_template,
                               handler=self.label_handler,
                               n_map_fields=self.n_map_fields,
                               sim_idx=sim_idx)

        # label_path = self.label_path_template.format(sim_idx=sim_idx)
        # label = self.label_handler.read(label_path)
        # label_tensor = torch.as_tensor(label)

        # feature_path_template = self.feature_path_template.format(sim_idx=sim_idx, freq="{freq}")
        # features = self.feature_handler.read(feature_path_template)
        features_tensor = tuple([torch.as_tensor(f) for f in features])
        features_tensor = torch.cat(features_tensor, dim=0)
        features_tensor = features_tensor.squeeze().transpose(1, 0)
        mean, std = features_tensor.mean(dim=0), features_tensor.std(dim=0)
        features_tensor = (features_tensor - mean) / std

        return features_tensor, label


class TestCMBMapDataset(Dataset):
    def __init__(self, 
                 n_sims,
                 freqs,
                 map_fields,
                #  label_path_template,
                #  label_handler,
                 feature_path_template,
                 feature_handler,
                 ):
        self.n_sims = n_sims
        self.freqs = freqs
        # self.label_path_template = label_path_template
        # self.label_handler = label_handler
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields:int = len(map_fields)

    def __len__(self):
        return self.n_sims
    
    def __getitem__(self, sim_idx):
        # label_path = self.label_path_template.format(sim_idx=sim_idx)
        # label = self.label_handler.read(label_path)
        # label_tensor = torch.as_tensor(label)

        features = _get_features_idx(freqs=self.freqs,
                                     path_template=self.feature_path_template,
                                     handler=self.feature_handler,
                                     n_map_fields=self.n_map_fields,
                                     sim_idx=sim_idx)


        # feature_path_template = self.feature_path_template.format(sim_idx=sim_idx, det="{det}")
        # features = self.feature_handler.read(feature_path_template)
        features_tensor = tuple([torch.as_tensor(f) for f in features])
        features_tensor = torch.cat(features_tensor, dim=0)
        features_tensor = features_tensor.squeeze().transpose(1, 0)
        mean, std = features_tensor.mean(dim=0), features_tensor.std(dim=0)
        features_tensor = (features_tensor - mean) / std
        
        return features_tensor, sim_idx
    

def _get_features_idx(freqs, path_template, handler, n_map_fields, sim_idx):
    features = []
    for freq in freqs:
        feature_path = path_template.format(sim_idx=sim_idx, freq=freq)
        feature_data = handler.read(feature_path)
        feature_data = reorder_map(feature_data, r2n=True)
        # Assume that we run either I or IQU
        features.append(feature_data[:n_map_fields, :])
    return features


def _get_label_idx(path_template, handler, n_map_fields, sim_idx):
    label_path = path_template.format(sim_idx=sim_idx)
    label = handler.read(label_path)
    label = reorder_map(label, r2n=True)
    if label.shape[0] == 3 and n_map_fields == 1:
        label = label[0, :]
    return label
