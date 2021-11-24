import torch
DEVICE = torch.device("cuda")
# BERT_MODEL_PATH = "distilbert-base-uncased"
BERT_MODEL_PATH = "bert-base-uncased"

MASK_TOKEN = 0
UNK_TOKEN = 1

CLAIM_ONLY = 1
EVIDENCE_ONLY = 2
CLAIM_AND_EVIDENCE = 3
INPUT_TYPE_ORDER = [None, "CLAIM_ONLY", "EVIDENCE_ONLY", "CLAIM_AND_EVIDENCE"]