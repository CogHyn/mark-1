import torch


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        
    def __call__(self, batch):
        videos = [item["video"]["pixel_values_videos"].squeeze(0) for item in batch]
        text = [item["text"] for item in batch]
        lsupport_frames = [item["support_frames"] for item in batch]

        collated_videos = torch.stack(videos)

        text_tokenized = self.tokenizer(text,
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt")

        collated_support_frames = torch.stack(lsupport_frames)
        
        return {
            "video" : {
                "pixel_values_videos": collated_videos
            },
            "text_input_ids" : text_tokenized["input_ids"],
            "text_attention_mask": text_tokenized["attention_mask"],
            "support_frames" : collated_support_frames
        }