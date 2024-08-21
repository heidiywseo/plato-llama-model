import torch
from tqdm import tqdm
from typing import List, Optional
import sentencepiece as spm
from model import llamaModel, ModelArgs


class TextGenerator:
    def __init__(self, model, tokenizer, max_seq_len: int = 512, max_batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.args = {
            "max_seq_len": max_seq_len,
            "max_batch_size": max_batch_size
        }

    def text_completion(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args['max_seq_len'] - 1

        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        batch_size = len(prompt_tokens)
        
        assert batch_size <= self.args['max_batch_size'], f"batch size must be less than or equal to {self.args['max_batch_size']}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        
        assert max_prompt_len <= self.args['max_seq_len'], f"prompt length must be less than or equal to {self.args['max_seq_len']}"
        total_len = min(self.args['max_seq_len'], max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.model.device)
        
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.model.device)
        
        eos_reached = torch.tensor([False] * batch_size, device=self.model.device)
        prompt_tokens_mask = tokens != pad_id  # True if the token is a prompt token, False otherwise
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        
        return (out_tokens, out_text)
    
    def _sample_top_p(self, probs, p):

        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)  # (B, vocab_size)

        probs_sum = torch.cumsum(probs_sort, dim=-1)  # (B, vocab_size)

        mask = probs_sum - probs_sort > p  # (B, vocab_size)

        probs_sort[mask] = 0.0 

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
    
if __name__ == "__main__":
    
        model_args = ModelArgs(
            dim=128,
            n_layers=2,
            n_heads=4,
            vocab_size=100, 
            max_seq_len=128,  
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        model = llamaModel(model_args).to(model_args.device)
        model.load_state_dict(torch.load('best_llama_model.pth',weights_only=True))
        tokenizer = spm.SentencePieceProcessor(model_file='tokenizer.model')  # Load or initialize your tokenizer

        text_gen = TextGenerator(model, tokenizer, max_seq_len=128, max_batch_size=32)
    
        prompts = ["Plato and life"]

        tokens, texts = text_gen.text_completion(prompts, temperature=0.7, top_p=0.9, max_gen_len=100)

        for i, text in enumerate(texts):
            print(f"Prompt {i+1}: {prompts[i]}")
            print(f"Generated: {text}")
            print("-" * 50)

