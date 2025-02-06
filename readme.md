# GenSAR: Unifying Balanced Search and Recommendation with Generative Retrieval
This is the official implementation of the paper "GenSAR: Unifying Balanced Search and Recommendation with Generative Retrieval" based on PyTorch.


# Quick Start


## Identifier

### Get Embedding

```bash
cd index/
python modules/get_emb.py
```

### Get Identifier

```bash
cd index/
python run_index.py
```


## Generative Model

### Doc2query

```bash
cd finetune/
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12345 extract_query.py
```


### Train the generative model

```bash
cd finetune/
python run_amazon.py
```