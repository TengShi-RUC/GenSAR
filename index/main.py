import argparse
import logging

from models.rqvae import RQVAE
from modules.dataset import EmbCollator, EmbDataset
from modules.trainer import Trainer
from modules.utils import setup_seed
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--seed', type=int, default=2024, help='random seed')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=5000,
                        help='number of epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=2048,
                        help='batch size')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
    )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner',
                        type=str,
                        default="AdamW",
                        help='optimizer')
    parser.add_argument('--lr_scheduler_type',
                        type=str,
                        default="constant",
                        help='scheduler')
    parser.add_argument('--warmup_epochs',
                        type=int,
                        default=50,
                        help='warmup epochs')

    parser.add_argument("--dataset", type=str, default="Amazon_Electronics")

    parser.add_argument("--semantic_emb", type=str, default="bge-base-zh-v1.5")

    parser.add_argument("--collaborate_emb",
                        type=str,
                        default="UniSAR_20241206-234322")

    parser.add_argument('--shared_e_dim',
                        type=int,
                        default=64,
                        help='vq codebook embedding size')
    parser.add_argument('--specific_e_dim',
                        type=int,
                        default=32,
                        help='vq codebook embedding size')

    parser.add_argument('--shared_num_emb_list',
                        type=int,
                        nargs='+',
                        default=[256, 256],
                        help='emb num of every vq')

    parser.add_argument('--semantic_num_emb_list',
                        type=int,
                        nargs='+',
                        default=[256, 256],
                        help='emb num of every vq')

    parser.add_argument('--collaborate_num_emb_list',
                        type=int,
                        nargs='+',
                        default=[256, 256],
                        help='emb num of every vq')

    parser.add_argument('--shared_sk_epsilons',
                        type=float,
                        nargs='+',
                        default=[0.0, 0.0],
                        help="sinkhorn epsilons")
    parser.add_argument('--specific_sk_epsilons',
                        type=float,
                        nargs='+',
                        default=[0.0, 0.003],
                        help="sinkhorn epsilons")

    parser.add_argument("--weight_decay",
                        type=float,
                        default=1e-5,
                        help='l2 regularization weight')
    parser.add_argument("--dropout_prob",
                        type=float,
                        default=0.0,
                        help="dropout ratio")
    parser.add_argument("--bn", type=int, default=0, help="use bn or not")
    parser.add_argument("--loss_type",
                        type=str,
                        default="mse",
                        help="loss_type")
    parser.add_argument("--kmeans_init",
                        type=int,
                        default=1,
                        help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters",
                        type=int,
                        default=100,
                        help="max kmeans iters")

    parser.add_argument("--sk_iters",
                        type=int,
                        default=50,
                        help="max sinkhorn iters")

    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="gpu or cpu")

    parser.add_argument('--quant_loss_weight',
                        type=float,
                        default=1.0,
                        help='vq quantion loss weight')
    parser.add_argument("--beta",
                        type=float,
                        default=0.25,
                        help="Beta for commitment loss")
    parser.add_argument('--layers',
                        type=int,
                        nargs='+',
                        default=[2048, 1024, 512, 256, 128, 64],
                        help='hidden sizes of every layer')

    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir",
                        type=str,
                        default="",
                        help="output directory for model")

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    args = parse_args()
    for flag, value in args.__dict__.items():
        logging.info('{}: {} {}'.format(flag, value, type(value)))

    setup_seed(args.seed)
    """build dataset"""
    data = EmbDataset(args)
    model = RQVAE(
        semantic_dim=data.semantic_dim,
        collaborate_dim=data.collaborate_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_iters=args.sk_iters,
        shared_n_e_list=args.shared_num_emb_list,
        semantic_n_e_list=args.semantic_num_emb_list,
        collaborate_n_e_list=args.collaborate_num_emb_list,
        shared_e_dim=args.shared_e_dim,
        specific_e_dim=args.specific_e_dim,
        shared_sk_epsilons=args.shared_sk_epsilons,
        specific_sk_epsilons=args.specific_sk_epsilons,
    )
    print(model)
    num_parameters = model.count_variables()
    logging.info("num model parameters:{}".format(num_parameters))
    data_loader = DataLoader(data,
                             num_workers=args.num_workers,
                             batch_size=args.batch_size,
                             shuffle=True,
                             pin_memory=True,
                             collate_fn=EmbCollator())
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)

    logging.info("Best Loss: {}".format(best_loss))
    logging.info("Best Collision Rate: {}".format(best_collision_rate))
