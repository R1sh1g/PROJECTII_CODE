# src/eval_maml.py

import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import MODEL_NAME, N_WAY, K_SHOT, Q_QUERY, INNER_LR, INNER_STEPS, DEVICE
from .dataloader import load_csv, build_index, valid_aspects
from .episodic_sampler import sample_episode
from .model import ACSCTransformer
from .utils import get_tokenizer, encode_batch
from .maml import get_adaptable_weights, inner_adapt, forward_with_weights

@torch.no_grad()
def acc_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean().item()

def main():
    device = torch.device(DEVICE)

    test_data = load_csv("C:\\Users\\Rishi\\PROJECTII_CODE\\models\\ASC\\maml\\testmaml.csv")
    test_index = build_index(test_data)

    aspects = valid_aspects(test_index, N_WAY, K_SHOT, Q_QUERY)
    if len(aspects) == 0:
        raise RuntimeError("No valid test aspects for episodic evaluation.")

    tokenizer = get_tokenizer(MODEL_NAME)
    model = ACSCTransformer(MODEL_NAME, num_labels=N_WAY).to(device)
    model.load_state_dict(torch.load("maml_acsc_epoch3.pt", map_location=device))
    model.eval()

    episodes = 200
    accs = []
    losses = []

    for _ in tqdm(range(episodes), desc="Evaluating"):
        asp = random.choice(aspects)
        support, query = sample_episode(test_index, asp, N_WAY, K_SHOT, Q_QUERY)

        support_batch = encode_batch(tokenizer, support, device)
        query_batch = encode_batch(tokenizer, query, device)

        # Inner-loop adaptation (needs grad), so temporarily enable grad
        with torch.enable_grad():
            weights = get_adaptable_weights(model)
            fast = inner_adapt(model, weights, support_batch, INNER_LR, INNER_STEPS)

        q_input, q_attn, q_labels = query_batch
        q_logits = forward_with_weights(model, q_input, q_attn, fast)
        q_loss = F.cross_entropy(q_logits, q_labels).item()

        losses.append(q_loss)
        accs.append(acc_from_logits(q_logits, q_labels))

    print(f"Episode Acc: {sum(accs)/len(accs):.4f}")
    print(f"Episode Loss: {sum(losses)/len(losses):.4f}")

if __name__ == "__main__":
    main()
