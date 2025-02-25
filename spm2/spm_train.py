#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# https://github.com/pytorch/fairseq/blob/master/LICENSE
import sys

import sentencepiece as spm


if __name__ == "__main__":
    spm.SentencePieceTrainer.Train(" ".join(sys.argv[1:]))

# python spm_train.py --input=input.txt --model_prefix=unigram5000 --vocab_size=5000 --character_coverage=1.0 --model_type=unigram