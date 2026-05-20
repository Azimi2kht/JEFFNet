import os
from pathlib import Path

import torch


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta

        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return True  # first is best

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

            return False


def make_embeddings(device, data_loader, encoder, epochs=1):
    ipe = len(data_loader)

    z_mem, l_mem = [], []

    for _ in range(epochs):
        for itr, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.to(device)
            with torch.no_grad():
                z = encoder(imgs)
                z = torch.mean(z, dim=1)
                z = z.cpu()
            labels = labels.cpu()
            z_mem.append(z)
            l_mem.append(labels)
            if itr % 50 == 0:
                print(f"[{itr}/{ipe}]")

    z_mem = torch.cat(z_mem, 0)
    l_mem = torch.cat(l_mem, 0)
    print(z_mem.shape)
    print(l_mem.shape)

    return z_mem, l_mem


def load_or_make_embeddings(
    embs_path,
    device,
    dataloader,
    encoder,
    logger,
    save_path=None,
):
    """
    Loads embeddings from disk if they exist.
    Otherwise, creates them using make_embeddings() and saves them.

    Returns:
        embs, labs
    """

    if embs_path and os.path.exists(embs_path):
        checkpoint = torch.load(embs_path, map_location="cpu")
        embs = checkpoint["embs"]
        labs = checkpoint["labs"]
        logger.info(f"Loaded embs of shape {embs.shape} from: {embs_path}")
    else:
        embs, labs = make_embeddings(device, dataloader, encoder)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"embs": embs, "labs": labs}, save_path)
            logger.info(f"Saved embs of shape {embs.shape} to: {save_path}")
        else:
            logger.info(f"Created embs of shape {embs.shape} (not saved)")

    return embs, labs
    return embs, labs
