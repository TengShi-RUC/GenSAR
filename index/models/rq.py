import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(
        self,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=100,
        sk_iters=100,
        shared_n_e_list=None,
        semantic_n_e_list=None,
        collaborate_n_e_list=None,
        shared_e_dim=None,
        specific_e_dim=None,
        shared_sk_epsilons=None,
        specific_sk_epsilons=None,
    ):
        super().__init__()
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_iters = sk_iters

        self.shared_n_e_list = shared_n_e_list
        self.semantic_n_e_list = semantic_n_e_list
        self.collaborate_n_e_list = collaborate_n_e_list

        self.shared_e_dim = shared_e_dim
        self.specific_e_dim = specific_e_dim
        assert self.shared_e_dim // 2 == self.specific_e_dim

        self.shared_sk_epsilons = shared_sk_epsilons
        self.specific_sk_epsilons = specific_sk_epsilons

        self.num_quantizers = len(shared_n_e_list) + len(semantic_n_e_list)
        assert len(semantic_n_e_list) == len(collaborate_n_e_list)

        self.vq_layers_shared = nn.ModuleList([
            VectorQuantizer(n_e,
                            shared_e_dim,
                            beta=self.beta,
                            kmeans_init=self.kmeans_init,
                            kmeans_iters=self.kmeans_iters,
                            sk_epsilon=sk_epsilon,
                            sk_iters=sk_iters)
            for n_e, sk_epsilon in zip(shared_n_e_list, shared_sk_epsilons)
        ])
        self.vq_layers_semantic = nn.ModuleList([
            VectorQuantizer(n_e,
                            specific_e_dim,
                            beta=self.beta,
                            kmeans_init=self.kmeans_init,
                            kmeans_iters=self.kmeans_iters,
                            sk_epsilon=sk_epsilon,
                            sk_iters=sk_iters)
            for n_e, sk_epsilon in zip(semantic_n_e_list, specific_sk_epsilons)
        ])
        self.vq_layers_collaborate = nn.ModuleList([
            VectorQuantizer(n_e,
                            specific_e_dim,
                            beta=self.beta,
                            kmeans_init=self.kmeans_init,
                            kmeans_iters=self.kmeans_iters,
                            sk_epsilon=sk_epsilon,
                            sk_iters=sk_iters) for n_e, sk_epsilon in zip(
                                collaborate_n_e_list, specific_sk_epsilons)
        ])

    def forward(self, x, use_sk=True):
        shared_losses = []
        shared_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers_shared:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            shared_losses.append(loss)
            shared_indices.append(indices)

        semantic_losses = []
        semantic_indices = []

        residual_semantic = residual[:, :self.specific_e_dim]
        semantic_x_q = x_q[:, :self.specific_e_dim]
        for quantizer in self.vq_layers_semantic:
            x_res, loss, indices = quantizer(residual_semantic, use_sk=use_sk)
            residual_semantic = residual_semantic - x_res
            semantic_x_q = semantic_x_q + x_res

            semantic_losses.append(loss)
            semantic_indices.append(indices)

        collaborate_losses = []
        collaborate_indices = []

        residual_collaborate = residual[:, self.specific_e_dim:]
        collaborate_x_q = x_q[:, self.specific_e_dim:]
        for quantizer in self.vq_layers_collaborate:
            x_res, loss, indices = quantizer(residual_collaborate,
                                             use_sk=use_sk)
            residual_collaborate = residual_collaborate - x_res
            collaborate_x_q = collaborate_x_q + x_res

            collaborate_losses.append(loss)
            collaborate_indices.append(indices)

        all_losses = shared_losses + semantic_losses + collaborate_losses
        mean_losses = torch.stack(all_losses).mean()

        semantic_indices = torch.stack(shared_indices + semantic_indices,
                                       dim=-1)
        collaborate_indices = torch.stack(shared_indices + collaborate_indices,
                                          dim=-1)

        return (semantic_x_q,
                collaborate_x_q), mean_losses, (semantic_indices,
                                                collaborate_indices)
