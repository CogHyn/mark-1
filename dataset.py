import numpy as np
import os
import torch

from torch.utils.data import Dataset
from utils import load_anno
from torchcodec.decoders import VideoDecoder
from typing import List



class VideoData(Dataset):
    def __init__(self, *, root_dir, train_dir, anno_file, processor, config):
        anno_path = os.path.join(root_dir, train_dir, anno_file)
        annotation_file = load_anno(anno_path)
        self._root_dir = root_dir
        self._train_dir = train_dir
        self._anno_file = anno_file
        self._count = annotation_file["__count__"]
        self._data = annotation_file["data"]
        self._processor = processor
        self.cache = {}
        self.config = config

    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        video_path = self._data[idx]["video_path"]
        support_frames = self._data[idx]["support_frames"]
        data_id = self._data[idx]["id"]
        question = self._data[idx]["question"]
        choices = self._data[idx]["choices"]
        answer = self._data[idx]["answer"]
        
        if video_path in self.cache.keys():
            video_processed = self.cache[video_path]["video_processed"]
            frames_indices = self.cache[video_path]["frame_indices"]
            
        else:
            vr = VideoDecoder(os.path.join(self._root_dir, video_path))
            frames_indices = self.sample_indices(len(vr), self.config["number_of_frames"])
            video = vr.get_frames_at(indices = frames_indices).data
            video_processed = self._processor(video, return_tensors="pt")
            
            self.cache[video_path] = {
                "video_processed" : video_processed,
                "frame_indices": frames_indices
            }

        processed_sf = np.array([0] * self.config["number_of_frames"])
        _sf = np.array(support_frames) * self.config["fps"]
        indices = abs(_sf[None, :] - np.array(frames_indices)[:, None])
        indices = np.argmin(indices, axis = 0)
        # print(_sf, frames_indices, indices)
        processed_sf[indices] = 1

        full_text_input = (
            question + "\n" +
            "Choices: " + ", ".join(choices) + 
            " <ANSWER> " + answer + 
            " <EOS>"  # Kết thúc chuỗi
        )
        
        return {
            "video": video_processed, 
            "text": full_text_input,
            "support_frames": torch.tensor(processed_sf)
        }
        
        
    @staticmethod
    def sample_indices(video_len: int, segment_len: int = 16) -> List[int]:
        rng = np.random.default_rng()
        segment_space = np.linspace(0, video_len - 1, segment_len + 1, dtype=int)

        frame_indices = []
        
        for i in range(1, len(segment_space)):
            start, end = segment_space[i - 1], segment_space[i]
            if start == end:
                frame_indices.append(start)
            else:
                idx = rng.integers(start, end)
                frame_indices.append(idx)
        return frame_indices