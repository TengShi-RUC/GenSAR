import argparse
import os
import pickle

import torch
from sentence_transformers import SentenceTransformer


def load_item_text(args):
    item_vocab_path = os.path.join("../data", args.dataset,
                                   "vocab/item_vocab.pkl")
    print("load item vocab: {}".format(item_vocab_path))

    item_vocab = pickle.load(open(item_vocab_path, "rb"))
    item_text = []
    for item_id in range(1, len(item_vocab)):
        cur_item_text = item_vocab[item_id]
        item_text.append(' '.join(
            [cur_item_text['title'], cur_item_text['description']]))

    return item_text


def load_text_encoder_sentence(args):
    model_path = os.path.join(args.llm_path, args.llm_name)
    print("load encoder: {}".format(model_path))

    model_path = os.path.join(model_path)
    model = SentenceTransformer(model_path)
    model.max_seq_length = args.max_sent_len
    print(model)
    return model


def generate_item_embedding_sentence(args, item_text_list,
                                     model: SentenceTransformer):
    print(f'Generate Text Embedding: ')
    print(' Dataset: ', args.dataset)

    batch_size = args.batch_size
    with torch.no_grad():

        embeddings = model.encode(item_text_list,
                                  batch_size=batch_size,
                                  show_progress_bar=True,
                                  convert_to_tensor=True,
                                  normalize_embeddings=True,
                                  device=args.device)
        embeddings = embeddings.cpu()
    print('Embeddings shape: ', embeddings.shape)

    save_path = os.path.join("../data", args.dataset, "emb")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, f"{args.llm_name}.pt")

    torch.save(embeddings, save_path)
    print("save embedding to: {}".format(save_path))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Amazon_Electronics')
parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')

parser.add_argument('--llm_name', type=str, default='bge-base-en-v1.5')
parser.add_argument('--llm_path', type=str, default='')

parser.add_argument('--max_sent_len', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=128)

if __name__ == '__main__':
    args = parser.parse_args()
    for flag, value in args.__dict__.items():
        print('{}: {}'.format(flag, value))

    device = torch.device(
        'cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    args.device = device

    item_text_list = load_item_text(args)

    model = load_text_encoder_sentence(args)
    generate_item_embedding_sentence(args, item_text_list, model)
