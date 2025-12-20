# Docs

## [DNA-BERT](DNA-BERT.pdf)

### Sequence Analysis

The sequence codes of DNA can only be correctly understood by looking at the context of the surrounding sequences -- usually, even very distant sequences have a semantic relationship with the target sequence.
This renders CNNs less effective, since they can only capture efficiently local patterns -- whose length is limited by the kernel size.
The same goes for RNNs, which in the long term show difficulties in capturing extremely long-range dependencies, both due to vanishing gradients and to the limited memory capacity of the hidden state.

This is the ideal scenario for the application of Transformer architectures.

### k-mers Tokenization

The tokenization of the DNA sequences is performed by splitting them into overlapping k-mers of different lenghts (the best are with k=6).
For example, with k=3, the sequence "ACGTGTA" would be tokenized as:

ACG, CGT, GTG, TGT, GTA

Given $k$, the vocabulary has size $4^k + 5$: they also add particular tokens, as **[CLS], [MASK], [PAD], [UNK], [SEP]**, respectively for classification, masking, padding, unknown k-mers and separation of sequences.

Each one of these k-mers is then treated as a single token, so it gets converted into an embedding vector. Moreover, to each embedding vector a positional encoding vector is added, to provide information about the position of the k-mer in the sequence.

### The Architecture

The model has input a sequence of k-mers:
$$
    x \in \mathbb{R}^{n \times d}
$$

with $d = 768$.

Then, the Encoder module is repeated $12$ times, each one composed of:
- Multi-Head Self-Attention with $12$ heads
- Layer Normalization and Residual Connections
- Feed-Forward Network with hidden size $3072$
- Layer Normalization and Residual Connections

At the end, the sequence is still of size:
$$
    y \in \mathbb{R}^{n \times d}
$$

### Pre-Training

The model is pre-trained on the Human Reference Genome, by **randomly** masking $15\%$ of the input k-mers and trying to predict them (Masked Language Modeling). Since masking random single k-mers would be too easy, they mask contiguous spans (of length $k$) of k-mers, so $k^2$ contiguous nucleotides are masked at once.
They employ the Cross-Entropy Loss.

### Fine-Tuning

The pre-trained model is then fine-tuned on specific tasks.