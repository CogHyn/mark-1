import torch
import numpy as np
from torchcodec.decoders import VideoDecoder 
import torch.nn.functional as F
from utils import load_model
import argparse

def get_frame_indices(video_len, num_frames=16):
    segment_space = np.linspace(0, video_len - 1, num_frames + 1, dtype=int)
    frame_indices = []
    rng = np.random.default_rng()
    
    for i in range(1, len(segment_space)):
        start, end = segment_space[i - 1], segment_space[i]
        if start == end:
            frame_indices.append(start)
        else:
            idx = rng.integers(start, end) 
            frame_indices.append(idx)
            
    return frame_indices


def predict(
    model,            
    processor,        
    tokenizer,        
    video_path, 
    question, 
    chat_history,
    num_frames=16,
    max_new_tokens=100,
    temperature=0.7, 
    top_p=0.9        
):
    if not video_path:
        return "", chat_history + [[question, "Pls Upload Video!"]]
    if not question:
        return "", chat_history

    print(f"Processing: {video_path}")
    print(f"Question: {question}")

    try:
        vr = VideoDecoder(video_path)
        total_frames = len(vr)
        indices = get_frame_indices(total_frames, num_frames)
        
        video_data = vr.get_frames_at(indices=indices).data
        video_inputs = processor(video_data, return_tensors="pt")
        
        device = model.device 
        video_input_dict = {
            "pixel_values_videos": video_inputs["pixel_values"].to(device)
        }

        prompt = f"{question} <ANSWER>"
        inputs = tokenizer(prompt, return_tensors="pt").to(device) 
        
        curr_input_ids = inputs.input_ids
        curr_att_mask = inputs.attention_mask
        
        generated_text = ""
        
        model.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                
                _, outputs = model(
                    video=video_input_dict, 
                    text_input_ids=curr_input_ids,
                    text_attention_mask=curr_att_mask
                )
                
                
                next_token_logits = outputs.logits[:, -1, :]
                
                next_token_logits = next_token_logits / temperature
                
                
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                
                sorted_indices_to_remove = cumulative_probs > top_p
                
                
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                
                
                next_token_logits[indices_to_remove] = float('-inf')

                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                

                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                word = tokenizer.decode(next_token.item())
                generated_text += word
                
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
                curr_att_mask = torch.cat([curr_att_mask, torch.ones((1, 1), device=device)], dim=1)


        chat_history.append((question, generated_text))
        return "", chat_history

    except Exception as e:
        print(f"Error: {e}")
        return "", chat_history + [[question, f"Error: {str(e)}"]]
    
if __name__ == "__main__":
    
    model, processor, tokenizer = load_model()
    
    parser = argparse.ArgumentParser(description="Mark-1 Inference CLI")
    parser.add_argument("--video", type=str, required=True, help="video path")
    parser.add_argument("--question", type=str, default="Mô tả video", help="User query")

    
    args = parser.parse_args()
    chat_history = []
    _, chat_history = predict(
        model,
        processor,
        tokenizer,
        args.video,
        args.question,
        chat_history
    )
    
    print("\n" + "="*30)
    print(f"MARK-1 SAYS:")
    print(chat_history[-1][1]) 
    print("="*30 + "\n")