"""Special-token ids for Panhapich/khmer-sp-8k. Fixed by that tokenizer; do not change."""

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
MASK_ID = 4

SPECIALS = {
    "<PAD>": PAD_ID,
    "<UNK>": UNK_ID,
    "<BOS>": BOS_ID,
    "<EOS>": EOS_ID,
    "<MASK>": MASK_ID,
}
