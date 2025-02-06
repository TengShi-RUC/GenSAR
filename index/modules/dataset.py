import logging
import os

import torch
import torch.nn.functional as F
import torch.utils.data as data


class EmbCollator:

    def __call__(self, batch):
        semantic_emb = torch.cat([x['semantic'].unsqueeze(0) for x in batch],
                                 dim=0)
        batch_size = len(semantic_emb)

        collaborate_emb = torch.cat(
            [x['collaborate'].unsqueeze(0) for x in batch], dim=0)

        return {
            "semantic": semantic_emb,
            "collaborate": collaborate_emb,
            "batch_size": batch_size
        }


class EmbDataset(data.Dataset):

    def __init__(self, args):
        self.semantic_path = os.path.join("../data", args.dataset, "emb",
                                          f"{args.semantic_emb}.pt")
        self.semantic_emb = torch.load(self.semantic_path,
                                       weights_only=True).float()
        self.semantic_dim = self.semantic_emb.shape[-1]
        logging.info("load semantic emb: {}".format(self.semantic_path))

        self.collaborate_path = os.path.join("../data", args.dataset, "emb",
                                             f"{args.collaborate_emb}.pt")
        self.collaborate_emb = torch.load(self.collaborate_path,
                                          weights_only=True).float()

        self.collaborate_dim = self.collaborate_emb.shape[-1]
        logging.info("load collaborate emb: {}".format(self.collaborate_path))

        assert len(self.semantic_emb) == len(self.collaborate_emb)

    def __getitem__(self, index):
        semantic_emb = self.semantic_emb[index]
        collaborate_emb = self.collaborate_emb[index]

        return {"semantic": semantic_emb, "collaborate": collaborate_emb}

    def __len__(self):
        return len(self.semantic_emb)
