# Appunti

---
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

---
## [DNA-BERT2](DNA-BERT2.pdf)

The work starts by underlying how DNA-BERT has some limitations, in particular in its $k$-mer tokenization strategy, due to the inefficiency of having valid and meaningful tokens and to the computational cost of having many overlapped tokens.

### Tokenization

Overlapping tokenization strategy - the one employed in [DNA-BERT](DNA-BERT.pdf) - has a few problems. The main one being data leakage: if we mask a stride of $k$ tokens in $k$-mer tokenization, we are giving the model the knowledge of the leftmost and righmost nucleotides of the masked region, which makes the prediction task easier, since the model has its search space reduced. If we mask just one token, the model can retrieve every nucleotide of the masked region.

Non-overlapping tokenization (proposed in [Nucelotide Transformer](nucleotide_transformer.pdf)) is instead plagued by sample inefficiency. In language tokenization, removing or editing a single token will, in general, not affect the tokenization of the rest of the sequence.

> For instance, given the sentence "The cat sat", removing the letter "T" leads to ["he", "cat", "sat"], affecting only the tokenization of the first token.

In non-overlapping tokenization, this is not true. Due to the small size of the DNA alphabet, removing or editing a single nucleotide will affect the tokenization of all the following tokens.

> Given the sequence "ACAATAATAATAATAACGG" - which gets tokenized as ["ACAATA", "ATAATA", "ATAACG", "G"] - removing the first "A" leads to "ACAATAATAATAATAACGG", which gets tokenized as ["CAATAA", "TAATAA", "TAACGG"], affecting all the tokens from then on.

This leads to the need for the model to align distinct representations of almost identical sequences, which is a difficult task.

#### Byte-Pair Encoding (BPE)

To address the above issues, DNA-BERT2 proposes a *Byte-Pair Encoding* (BPE) tokenization strategy. The technique helps avoiding information leakage, reducing the sequence length by almost $5x$ and by solving the sample inefficiency problem of non-overlapping tokenization. Finally, the number of nucleotides inside a token varies: this requires the model to understand both the number of nucleotides and the nucleotides themselves.

The algorithm works as follows:
1. Set the vocabulary to the unique letters of the corpus (in this case, the 4 nucleotides: A, C, G, T).
2. For each iteration:
   - Count all the adjacent symbol pairs in the corpus.
   - Find the most frequent pair.
   - Merge it into a new symbol and add it to the vocabulary.
3. Return the vocabulary.

> For instance, starting from the sequence "ACACAGTGTGT", we will have during the iterations the following:
> 1. Text: "A C A C A G T G T G T"; Vocabulary: {A, C, G, T}
> 2. Most frequent pair: "G T"; Text: "A C A C A GT GT GT"; Vocabulary: {A, C, G, T, GT}
> 3. Most frequent pair: "A C"; Text: "AC AC A GT GT GT"; Vocabulary: {A, C, G, T, GT, AC}
> ...and so on.

The vocabulary size is a hyperparameter to be set before BPE. The larger the vocabulary, of course, the less tokens in a single sequence; however, the more difficult it is for the model to learn good representations for all the tokens - since each token is used less frequently. In DNA-BERT2, they set the vocabulary size to $4,096$.

### The Architecture

The architecture starts from the classical Transformer Encoder. The main differences are:

1. Attention with Linear Biases (ALiBi); replacing learned positional embeddings.
2. Utilizing FlashAttention.
3. Employing LoRA in the fine-tuning phase.

#### Attention with Linear Biases (ALiBi)

The main issue with existing positional embeddings technique is the fact that whenever a model sees an input longer than the ones seen during training, it suffers from poor *extrapolation* capabilities. In order to solve this, the model applies some position-dependent biases to the attention scores and not to the input embeddings.

Given $q_i$ the $i$-th query vector, given $L$ the input length and $\mathbf{K}$ the key matrix, the attention scores become:
$$
    \text{softmax}\left(q_iK + m[-(i - 1), ..., -2, -1, 0, -1, -2, ..., -(L-1-i)]\right)
$$
with $m$ a fixed parameter *per-head*. Intuitively, ALiBi discourages attentions scores to very distant tokens, without any learning necessary.

---
## [Nucleotide Transformer](nucleotide_transformer.pdf)

### Model

Standard Transformer Encoder. Employs $6$-mers for tokens and learned positional embeddings (the NT-v2 uses rotary positional embeddings). The input is passed to a stack of Encoder Blocks, and finally to an LM head for predicting the Masked Tokens.

#### Encoder Block

1. LayerNorm, followed by Multi-Head Self-Attention.
2. Skip Connection with input.
3. LayerNorm, followed by a MLP of the form: GELU -> Linear -> GELU -> Linear.

### Training

Just like [DNA-BERT](DNA-BERT.pdf), from each input sequence a subset of $15\%$ of tokens is sampled. Inside the subset:

- $80\%$ gets masked (i.e., replaced with [MASK]);
- $10\%$ gets replaced with a random 'standard' token (no [CLS], [MASK] or [PAD]);

---
## [EVO](Evo.pdf)

---
## [EVO2](Evo2.pdf)

---
## [HyenaDNA](HyenaDNA.pdf)

---
## [JanusDNA](JanusDNA.pdf)

---
## [NucEL](NucEL.pdf)

---
## Global Info

### Benchmarks

Visto che l'encoder di per sè non fa nulla se non imparare una rappresentazione abbastanza robusta dei dati, bisogna poi usare benchmark che valutare la capacità del modello.

Quelli che ho trovato e che possono servirci sono:

- **Genome Understanding Evaluation (GUE)**: Una raccolta di una serie di dataset multi-specie, con task come Classificazione di Specie (es. per i funghi), Predizione del Fattore di Trascrizione, Detection dei Promoter e il mio preferito: Classificazione delle Varianti del Covid.