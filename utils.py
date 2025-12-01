from typing import Dict, Any 
import torch
import json 
import os
from sklearn.metrics import f1_score
import yaml
from model import VideoLLM
from transformers import AutoVideoProcessor, AutoTokenizer


def load_model():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model = VideoLLM(**config["model"])
    from huggingface_hub import hf_hub_download

    FILENAME = "checkpoint_last_v3.pt" 
    REPO_ID = "CogHyn/VideoLLM"

    local_checkpoint_path = hf_hub_download(
        repo_id=REPO_ID, 
        filename=FILENAME,
        repo_type="model" 
    )
    print(f"✅ Checkpoint downloaded to: {local_checkpoint_path}")
    checkpoint = torch.load(local_checkpoint_path, map_location='cpu') 
    model.projection.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.dispatch()    
    
    video_processor = AutoVideoProcessor.from_pretrained(config["model"]["video_repo"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["llm_repo"]) 
    
    return model, video_processor, tokenizer

def calculate_metrics(text_logits, text_labels, temp_logits, temp_labels):
    metrics = {}
    
    text_preds = torch.argmax(text_logits, dim=-1)
    
    mask = text_labels != -100
    correct = (text_preds[mask] == text_labels[mask]).sum()
    total = mask.sum()
    metrics['text_acc'] = (correct / total).item()
    
    temp_preds = (torch.sigmoid(temp_logits) > 0.5).long()
    
    y_true = temp_labels.detach().cpu().numpy().flatten()
    y_pred = temp_preds.detach().cpu().numpy().flatten()
    
    metrics['temp_f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    metrics['pred_positive_rate'] = y_pred.sum() / len(y_pred)
    
    return metrics


def load_anno(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        result = json.load(f)
    return result


def save_checkpoint_to_hub(api, local_path, remote_path, repo_id):
    print(f"Uploading {local_path} to Hugging Face Hub...")
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print("Upload done!")
    except Exception as e:
        print(f"Upload failed: {e}")
        
def save_checkpoint(api, model, optimizer, scheduler, epoch, step, loss, is_best = False, checkpoint_dir="checkpoints", version="v1"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainable_state_dict = model.projection.state_dict()
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': trainable_state_dict, 
        'optimizer_state_dict': optimizer.state_dict(), 
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss
    }

    latest_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    torch.save(checkpoint, latest_path)
    save_checkpoint_to_hub(api, latest_path, f"checkpoint_last_{version}.pt")
    print(f"Saved latest checkpoint to {latest_path}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)
        save_checkpoint_to_hub(api, best_path, f"checkpoint_best_{version}.pt")
        print(f"Saved BEST checkpoint (Loss: {loss:.4f}) to {best_path}")


def resume_checkpoint(model, optimizer, scheduler, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    
    model.projection.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    start_epoch = checkpoint['epoch'] + 1
    start_step = checkpoint['step']
    print(f"Resumed training from Epoch {start_epoch}, Step {start_step}")
    
    return start_epoch, start_step
