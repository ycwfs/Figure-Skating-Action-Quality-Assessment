import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations

@register_dataset('finefs')
class FineFS(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        vid_feat_folder,      # folder for features
        aud_feat_folder,
        annotation_folder,        # json file for annotations
        element_numbers,
        max_score,
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,        # input feat dim
        num_classes,      # number of action categories
        class_path,       # path to class label json file
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        force_upsampling  # force to upsample to max_seq_len
    ):

        # file path
        assert os.path.exists(vid_feat_folder) and os.path.exists(aud_feat_folder) and os.path.exists(annotation_folder)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.vid_feat_folder = vid_feat_folder
        self.aud_feat_folder = aud_feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.annotation_folder = annotation_folder

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling
        self.classes = json.load(open(class_path, 'r'))
        self.element_numbers = element_numbers
        print(self.classes)
        print(f"element numbers:{self.element_numbers}")

        # split / training mode
        self.split = split
        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio
        self.max_score = max_score

        self.is_training = is_training
        # load database and select the subset
        if is_training:
            self.dict_list = self._load_json_db(os.path.join(annotation_folder, 'annotation'))
        else:
            self.dict_list = self._load_json_db(os.path.join(annotation_folder, 'val_annotation'))
        # proposal vs action categories
        # assert (num_classes == 1) or (len(label_dict) == num_classes)

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'FineFS',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            'empty_label_ids': []
        }

    def get_attributes(self):
        return self.db_attributes

    def __len__(self):
        return len(self.dict_list)

    def convert_timestamp(self, time_str: str):
        time_parts = time_str.split(',')
        
        seconds_list = []
        
        for part in time_parts:
            minutes, seconds = map(int, part.split('-'))
            total_seconds = minutes * 60 + seconds
            seconds_list.append(total_seconds)
        
        return seconds_list


    def _process_elements(self, file_name, elements):
        segments, labels, element_scores = [], [], []
        for element in list(elements.values()):
            segments.append(self.convert_timestamp(element['time']))
            labels.append(self.classes[f"{element[f'{self.num_classes}_class']}"]) # xx element  coarse_class
            element_scores.append(round((element['score_of_pannel'] / self.max_score), 2))
        
        # load video,audio features
        feats = torch.from_numpy(np.load(os.path.join(self.vid_feat_folder, file_name + '_flow.npy'))).transpose(0, 1).float()
        audio_feats = torch.from_numpy(np.load(os.path.join(self.aud_feat_folder, file_name + '_vggish.npy'))).transpose(0, 1).float()
        vl = feats.shape[1]; al = audio_feats.shape[1]
        if vl > al:
            feats = feats[:, :al]
        elif al > vl:
            audio_feats = audio_feats[:, :vl]
        return {
            'video_id': file_name,
            'duration': feats.shape[1],
            'fps': torch.tensor(self.default_fps),
            'feats': feats,
            'audio_feats': audio_feats,
            'labels': torch.tensor(labels),
            'segments': torch.tensor(segments),
            'element_scores': torch.tensor(element_scores),
            'feat_stride': torch.tensor(self.feat_stride),
            'feat_num_frames': torch.tensor(self.num_frames)
        }

    def _load_json_db(self, annotation_folder):
        dict_list = []
        # loop the annotation folder to get the json file
        for file in os.listdir(annotation_folder):
            if file.endswith('.json'):
                file_name = file.split('.')[0]
                with open(os.path.join(annotation_folder, file), 'r') as f:
                    data = json.load(f)
                pcs = torch.tensor(round(data["total_program_component_score(factored)"]/100,2))
                elements = data['executed_element']
                en = len(elements)
                annotation_data = self._process_elements(file_name, elements)
                annotation_data['pcs'] = pcs
                dict_list.append(annotation_data)
        return dict_list


    def __getitem__(self, index):
        video_item = self.dict_list[index]
        return video_item
                        
