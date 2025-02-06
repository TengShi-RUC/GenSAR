import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):

    def __init__(
        self,
        semantic_dim=None,
        collaborate_dim=None,
        layers=None,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
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
        super(RQVAE, self).__init__()
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_iters = sk_iters

        self.semantic_dim = semantic_dim
        self.collaborate_dim = collaborate_dim

        self.shared_n_e_list = shared_n_e_list
        self.semantic_n_e_list = semantic_n_e_list
        self.collaborate_n_e_list = collaborate_n_e_list

        self.shared_e_dim = shared_e_dim
        self.specific_e_dim = specific_e_dim

        self.shared_sk_epsilons = shared_sk_epsilons
        self.specific_sk_epsilons = specific_sk_epsilons

        assert self.semantic_dim and self.collaborate_dim
        assert self.shared_e_dim // 2 == self.specific_e_dim

        self.hidden_dim = self.shared_e_dim // 2

        self.semantic_encode_layer_dims = [self.semantic_dim
                                           ] + self.layers + [self.hidden_dim]
        self.semantic_encoder = MLPLayers(
            layers=self.semantic_encode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn)

        self.semantic_decoder_layer_dims = self.semantic_encode_layer_dims[::
                                                                           -1]
        self.semantic_decoder = MLPLayers(
            layers=self.semantic_decoder_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn)

        self.collaborate_encode_layer_dims = [
            self.collaborate_dim
        ] + self.layers + [self.hidden_dim]
        self.collaborate_encoder = MLPLayers(
            layers=self.collaborate_encode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn)

        self.collaborate_decoder_layer_dims = self.collaborate_encode_layer_dims[::
                                                                                 -1]
        self.collaborate_decoder = MLPLayers(
            layers=self.collaborate_decoder_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn)

        self.rq = ResidualVectorQuantizer(
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_iters=self.sk_iters,
            shared_n_e_list=self.shared_n_e_list,
            semantic_n_e_list=self.semantic_n_e_list,
            collaborate_n_e_list=self.collaborate_n_e_list,
            shared_e_dim=self.shared_e_dim,
            specific_e_dim=self.specific_e_dim,
            shared_sk_epsilons=self.shared_sk_epsilons,
            specific_sk_epsilons=self.specific_sk_epsilons,
        )

    def forward(self, x, use_sk=True):
        semantic_x = self.semantic_encoder(x['semantic'])
        collaborate_x = self.collaborate_encoder(x['collaborate'])
        x = torch.cat([semantic_x, collaborate_x], dim=-1)

        x_q, rq_loss, indices = self.rq(x, use_sk=use_sk)

        semantic_x_q, collaborate_x_q = x_q
        semantic_x_q = self.semantic_decoder(semantic_x_q)
        collaborate_x_q = self.collaborate_decoder(collaborate_x_q)

        out = {"semantic": semantic_x_q, "collaborate": collaborate_x_q}

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, x, use_sk=False):
        semantic_x = self.semantic_encoder(x['semantic'])
        collaborate_x = self.collaborate_encoder(x['collaborate'])
        x = torch.cat([semantic_x, collaborate_x], dim=-1)

        _, _, indices = self.rq(x, use_sk=use_sk)

        return indices

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_fn = F.mse_loss
        elif self.loss_type == 'l1':
            loss_fn = F.l1_loss
        else:
            raise ValueError('incompatible loss type')

        semantic_loss = loss_fn(out['semantic'],
                                xs['semantic'],
                                reduction='mean')

        collaborate_loss = loss_fn(out['collaborate'],
                                   xs['collaborate'],
                                   reduction='mean')

        loss_recon = semantic_loss + collaborate_loss

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon, quant_loss

    def count_variables(self) -> int:
        total_parameters = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                num_p = p.numel()
                total_parameters += num_p

        return total_parameters
