from dmtools.common import *


@pysnooper.snoop()  #函数细节执行时间和过程
def number_to_bits(number):
    if number:
        bits = []
        while number:
            number, remainder = divmod(number, 2)
            bits.insert(0, remainder)
        return bits
    else:
        return [0]

# number_to_bits(6)

"""
What is PyTorch-Transformers?

I have taken this section from PyTorch-Transformers’ documentation. This library currently contains PyTorch implementations, pre-trained model weights, usage scripts and conversion utilities for the following models:

    BERT (from Google) released with the paper BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    GPT (from OpenAI) released with the paper Improving Language Understanding by Generative Pre-Training
    GPT-2 (from OpenAI) released with the paper Language Models are Unsupervised Multitask Learners
    Transformer-XL (from Google/CMU) released with the paper Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
    XLNet (from Google/CMU) released with the paper XLNet: Generalized Autoregressive Pretraining for Language Understanding
    XLM (from Facebook) released together with the paper Cross-lingual Language Model Pretraining
"""

"""
Fast Neptune库能够快速记录开展机器学习测试所需的所有信息。
"""
