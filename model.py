import torch 
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, AutoModel,
    BertConfig, BertLayer
)

class DeviceAwareModule(nn.Module):
    """
    """
    @property
    def device(self):

        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def debug_shape(self, batch_size = 1):
        return torch.randn(batch_size, 1, 1).to(self.device)
    
    
class ResidualBlock(nn.Module):
    """
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)
    
    


class Projection(DeviceAwareModule):
    def __init__(self, *, 
                 num_visual_token, 
                 num_temporal_token, 
                 video_seq_len, 
                 embed_dim_video, 
                 text_seq_len, 
                 embed_dim_text, 
                 num_heads,
                 num_res_blocks,
                 bert_config, 
                 number_of_frames):
        super().__init__()

        self.nqueries = num_visual_token
        self.nregression = num_temporal_token
        self.lseq_video = video_seq_len
        self.d_video = embed_dim_video
        self.lseq_text = text_seq_len
        self.d_text = embed_dim_text
        self.video_projection = nn.Linear(embed_dim_video, embed_dim_text)
        
        self.query_tokens = nn.Parameter(torch.randn(1, num_visual_token + num_temporal_token, embed_dim_text))
        nn.init.orthogonal_(self.query_tokens)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim_text, num_heads=num_heads, batch_first=True)
        
        layers = []

        layers.append(nn.Linear(embed_dim_text, embed_dim_text))
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(embed_dim_text, dropout=0.1))
        layers.append(nn.LayerNorm(embed_dim_text))
        layers.append(nn.Linear(embed_dim_text, number_of_frames))
        self.reg_layer = nn.Sequential(*layers)
        
        self.bert_config = BertConfig(**bert_config)
        self.bert_layer = BertLayer(self.bert_config)


    def forward(self, video, questions_embed, questions_attention_mask):
        batch_size = video.shape[0]
        video = video.to(self.device)
        questions_embed = questions_embed.to(self.device)
        questions_attention_mask = questions_attention_mask.to(self.device)

        video = self.video_projection(video)
        learnable_queries = self.query_tokens.expand(batch_size, -1, -1)

        visual_latents, _ = self.cross_attention(
            query=learnable_queries,
            key=video,
            value=video
        )

        

        combined_features = torch.cat([visual_latents, questions_embed], dim=1)
        query_mask = torch.ones(batch_size, self.nqueries + self.nregression).to(video.device)
        combined_mask = torch.cat([query_mask, questions_attention_mask], dim=1)

        extended_attention_mask = combined_mask[:, None, None, :]
        
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        full_output = self.bert_layer(combined_features, attention_mask=extended_attention_mask)[0]

        contextualized_queries = full_output[:, :self.nqueries + self.nregression, :]

        llm_visual_prompts = contextualized_queries[:, :self.nqueries, :]
        localization_tokens = contextualized_queries[:, self.nqueries:, :]
        
        raw_mask_predictions = self.reg_layer(localization_tokens)
        temporal_saliency_map, _ = raw_mask_predictions.max(dim=1)

        return llm_visual_prompts, temporal_saliency_map
        
        
    


class VideoLLM(DeviceAwareModule):
    def __init__(self, *, video_repo, llm_repo,
                 projection_config,
                 device_map,
                 **kwargs
                ):
        super().__init__()
        self.video_encoder = AutoModel.from_pretrained(video_repo)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_repo)
        self.device_map = device_map

        for param in self.video_decoder.parameters():
            param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False

        
        # video decoder (B, L, 1024) -> (B, 12, 1024)
        self.projection = Projection(**projection_config)

    def dispatch(self):
        self.video_decoder.to(self.device_map["video"])
        self.projection.to(self.device_map["projection"])
        self.llm_model.to(self.device_map["llm"])


    def freeze(self):
        for param in self.video_decoder.parameters():
            param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False    
    
        
    def forward(self, *, 
               video,
               text_input_ids,
               text_attention_mask,
               **kwargs):

        video["pixel_values_videos"] = video["pixel_values_videos"].to(self.video_decoder.device)
        text_input_ids = text_input_ids.to(self.llm_model.device)
        text_attention_mask = text_attention_mask.to(self.llm_model.device)
        
        video_embed = self.video_encoder.get_vision_features(**video)
        text_embeds = self.llm_model.get_input_embeddings()(text_input_ids)
        
        projection, temporal = self.projection(video_embed, text_embeds, text_attention_mask)
        
        
        projection = projection.to(self.llm_model.device)
        temporal = temporal.to(self.llm_model.device)
        text_embeds = text_embeds.to(self.llm_model.device)
        text_attention_mask = text_attention_mask.to(self.llm_model.device)
        
        fused_embeds = torch.cat([projection, text_embeds], dim=1)
        L_video = projection.shape[1] 
        B = projection.shape[0]

        video_mask = torch.ones(
            B, 
            L_video, 
            dtype=text_attention_mask.dtype, 
            device=self.llm_model.device 
        )

        fused_attention_mask = torch.cat(
            [video_mask, text_attention_mask], 
            dim=1 
        )

        outputs = self.llm_model(
            inputs_embeds=fused_embeds,
            attention_mask=fused_attention_mask,
        )
        return temporal, outputs

        

        

        