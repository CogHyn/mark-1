from dataset import VideoData
from collator import DataCollator
from model import VideoLLM

import torch

from transformers import AutoVideoProcessor, AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from utils import *

from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi, login
import torch

import os
from dotenv import load_dotenv
import wandb
from tqdm import tqdm
import yaml


FILENAME = "..."
REPO_ID = "..."
VERSION = "..."


def train(**cfg):
    video_processor = AutoVideoProcessor.from_pretrained(cfg["model"]["video_repo"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["llm_repo"])
    
    video_dataset = VideoData(**cfg["training"]["dataset"], 
                        processor = video_processor)
    custom_collate_fn = DataCollator(tokenizer=tokenizer)
    
    train_loader = DataLoader(video_dataset, 
                        **cfg["training"]["dataloader"],
                        collate_fn=custom_collate_fn,
    )
    
    num_epochs = cfg["training"]["num_epochs"]
    warmup_ratio = cfg["training"]["warmup_ratio"]
    lr = cfg["training"]["lr"]
    weight_decay = cfg["training"]["weight_decay"]
    pos_weight_value = cfg["training"]["pos_weight"]
    lambda_temp = cfg["training"]["lambda_temp"]
    DEVICE_MAP = cfg["model"]["device_map"]
    
    # Set up model
    model = VideoLLM(**cfg["model"])
    model.dispatch()
    
    print(f"Video Encoder: {model.video_encoder.device}")
    print(f"Projection: {model.projection.device}")
    print(f"LLM: {model.llm_model.device}")
    
    model.freeze()
    
    len_train_loader = len(train_loader)
    
    num_training_steps = num_epochs * len_train_loader 

    num_warmup_steps = int(num_training_steps * warmup_ratio)
    

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    scaler = torch.amp.GradScaler('cuda')
    loss_fct_text = nn.CrossEntropyLoss(ignore_index=-100) 
    pos_weight_tensor = torch.tensor([pos_weight_value]).to(DEVICE_MAP["llm"]) 
    loss_fct_temp = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    
    best_text_acc = 0.0

    
    api = HfApi()
    local_checkpoint_path = hf_hub_download(
        repo_id=REPO_ID, 
        filename=FILENAME,
        repo_type="model" 
    )
    print(f"✅ Checkpoint downloaded to: {local_checkpoint_path}")
    start_epoch = 0
    start_epoch, start_step = resume_checkpoint(model, optimizer, scheduler, local_checkpoint_path)
    print(f"Sucessfully resume: {start_epoch}, {start_step}")
    global_step = start_step


    model.train() 
    try:
        for epoch in range(start_epoch, num_epochs):
            
            print(f"Epoch {epoch + 1}/{num_epochs}")
            progress_bar = tqdm(train_loader)
            
            total_loss = 0
            total_correct_text = 0
            total_text_tokens = 0
            
            for batch in progress_bar:
                
                
                labels = batch['text_input_ids'].to(DEVICE_MAP["llm"]) 
                
                temporal_gt = batch['support_frames'].to(DEVICE_MAP["llm"]).float()
        
                optimizer.zero_grad()
        
                
                with torch.amp.autocast('cuda', dtype=torch.float16): 
                    
                    
                    pred_temporal, llm_outputs = model(
                        **batch
                    )
                    
                    loss_temp = loss_fct_temp(pred_temporal, temporal_gt)
                    
                    
                    logits = llm_outputs.logits 
                    
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    seq_len_text = shift_labels.shape[1]
                    shift_logits_text_only = shift_logits[:, -seq_len_text:, :]
                    
                    loss_text = loss_fct_text(
                        shift_logits_text_only.reshape(-1, shift_logits_text_only.size(-1)), 
                        shift_labels.reshape(-1)
                    )
        
                    loss = loss_text + lambda_temp * loss_temp
        
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1


                with torch.no_grad():
                    text_preds = torch.argmax(shift_logits_text_only, dim=-1)
                    mask = shift_labels != -100 
                    correct = (text_preds[mask] == shift_labels[mask]).sum().item()
                    total = mask.sum().item()
                    
                    total_correct_text += correct
                    total_text_tokens += total
                
                if global_step % 50 == 0:
                    with torch.no_grad():
                        mtr = calculate_metrics(shift_logits_text_only, shift_labels, pred_temporal, temporal_gt)
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/text_acc": mtr['text_acc'],
                        "train/temp_f1": mtr['temp_f1'],
                        "train/lr": scheduler.get_last_lr()[0],
                        "global_step": global_step
                    })
                current_batch_acc = correct / total if total > 0 else 0
                total_loss += loss.item()
                progress_bar.set_description(f"Loss: {loss.item():.3f} | Acc: {current_batch_acc:.2%}")
            avg_loss = (total_loss / len(train_loader)) 
            epoch_text_acc = total_correct_text / total_text_tokens if total_text_tokens > 0 else 0.0
            is_best = epoch_text_acc > best_text_acc
            
            if is_best:
                best_text_acc = epoch_text_acc
                print(f"New Best Accuracy: {best_text_acc:.2%}")

            save_checkpoint(api, model, optimizer, scheduler, epoch, global_step, avg_loss, is_best=is_best)
            print(f"Average Loss: {avg_loss}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user! Saving emergency checkpoint...")
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss.item())
        print("Done. Exiting safe.")
    
    
    
    


if __name__ == "__main__":
    
    load_dotenv()
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print(config)
    
    VERSION = config["version"]
    
    hf_token = os.environ["HF_TOKEN"]
    wandb_token = os.environ["WANDB"]
    
    login(token=hf_token)
    wandb.login(key=wandb_token)
    wandb.init(
        project="VideoLLM-Mark1",
        name=f"Run-{VERSION}-BERT-MaxPooling",
        config=config["training"]
    )
    
    print("Start training ...")
    
    train(**config)
    
    print("End training ...")
    
    
    
    

    
    