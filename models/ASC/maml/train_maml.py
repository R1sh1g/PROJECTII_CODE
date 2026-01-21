# src/train_maml.py

import random
import torch
from tqdm import tqdm

from .config import (
    MODEL_NAME, N_WAY, K_SHOT, Q_QUERY, TASKS_PER_META_BATCH,
    INNER_LR, INNER_STEPS, META_LR, EPOCHS, DEVICE
)
from .dataloader import load_csv, build_index, valid_aspects
from .episodic_sampler import sample_episode
from .model import ACSCTransformer
from .utils import get_tokenizer, encode_batch
from .maml import get_adaptable_weights, inner_adapt, query_loss

def main():
    device = torch.device(DEVICE)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    train_data = load_csv("C:\\Users\\Rishi\\PROJECTII_CODE\\models\\ASC\\maml\\trainmaml.csv")
    train_index = build_index(train_data)

    aspects = valid_aspects(train_index, N_WAY, K_SHOT, Q_QUERY)
    if len(aspects) == 0:
        raise RuntimeError("No valid aspects found with enough samples for episode construction.")

    tokenizer = get_tokenizer(MODEL_NAME)
    model = ACSCTransformer(MODEL_NAME, num_labels=N_WAY).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=META_LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(range(200), desc=f"Epoch {epoch}")  # 200 meta-iterations per epoch (tune as needed)

        for _ in pbar:
            opt.zero_grad()
            meta_loss = 0.0

            for _task in range(TASKS_PER_META_BATCH):
                asp = random.choice(aspects)
                support, query = sample_episode(train_index, asp, N_WAY, K_SHOT, Q_QUERY)

                support_batch = encode_batch(tokenizer, support, device)
                query_batch = encode_batch(tokenizer, query, device)

                weights = get_adaptable_weights(model)
                fast = inner_adapt(model, weights, support_batch, INNER_LR, INNER_STEPS)

                q_loss = query_loss(model, fast, query_batch)
                meta_loss += q_loss

            meta_loss = meta_loss / TASKS_PER_META_BATCH
            meta_loss.backward()
            opt.step()

            pbar.set_postfix({"meta_loss": float(meta_loss.detach().cpu())})

        # Save checkpoint each epoch
        torch.save(model.state_dict(), f"maml_acsc_epoch{epoch}.pt")

if __name__ == "__main__":
    main()
