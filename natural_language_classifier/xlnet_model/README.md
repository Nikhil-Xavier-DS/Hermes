# [RoBERTa based Text Classification Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_classifier/roberta_model)
The code is implemented based on the publication, [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). 
RoBERTa is built on BERT and modified key hyperparameters and removes the next-sentence pretraining objective, and also training with much larger mini-batches and learning rates.

Language model pre-training has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, the paper shows that hyperparameter choices have significant impact on the final results. The paper shows that BERT was significantly undertrained, and can match or exceed the performance of every model published after it. RoBERTa models achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight the importance of previously overlooked design choices.

RoBERTa shows that performance can be substantially improved by training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data. 

<img src="https://pytorch.org/tutorials/_images/bert.png" width="360">

RoBERTa has the same architecture as BERT, but uses a byte-level BPE as a tokenizer (same as GPT-2) and uses a different pretraining scheme.

#### Reference
1. Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. 2019. "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
2. https://huggingface.co/transformers/index.html
