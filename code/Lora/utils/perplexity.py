"""WikiText-2 perplexity computation for retention testing."""

import torch
from datasets import load_dataset
from tqdm import tqdm


def compute_wikitext2_ppl(model, tokenizer, stride=512, max_length=1024, device="cuda"):
    """Compute sliding-window perplexity on WikiText-2 test set."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end = 0

    for begin in tqdm(range(0, seq_len, stride), desc="WikiText-2 PPL"):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end

        input_ids = encodings.input_ids[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * target_len)

        prev_end = end
        if end == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / prev_end)
    return ppl.item()
