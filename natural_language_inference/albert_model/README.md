# [ALBERT based Natural Language Inference Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_inference/albert_model)
The code is implemented based on the publication, [ ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942). The paper proposes two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT:
a) Splitting the embedding matrix into two smaller matrices.
b) Using repeating layers split among groups.

Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations, longer training times, and unexpected model degradation. To address these problems, the paper proposes a two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows that the proposed methods lead to models that scale much better compared to the original BERT.
  
Paper also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, the ALBERT model establishes new state-of-the-art results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large.

<img src="https://pytorch.org/tutorials/_images/bert.png" width="360">

ALBERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left. It also uses repeating layers which results in a small memory footprint, however the computational cost remains similar to a BERT-like architecture with the same number of hidden layers as it has to iterate through the same number of (repeating) layers.

#### Reference
1. Zhenzhong Lan1 Mingda Chen2∗ Sebastian Goodman1 Kevin Gimpel2 Piyush Sharma1 Radu Soricut. 2019. "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"
2. https://huggingface.co/transformers/index.html
