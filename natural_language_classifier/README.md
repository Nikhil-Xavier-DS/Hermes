# Natural Language Classification
Text classification is the task of assigning a set of predefined categories to sentences. Text classification in Hermes deals with Sentiment analysis. Sentiment analysis  is a natural language processing technique used to interpret and classify emotions in subjective data. For sentiment analysis, the algorithms use IMDB dataset.

IMDB is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. It provides a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. 

Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human languages, in particular, programming computers to process and analyze large amounts of natural language data. Deep neural network-style machine learning methods are widespread in natural language processing and can achieve state-of-the-art results in many natural language tasks. 

Some of the popular text classification (sentiment analysis in our case) algorithms implemented in Hermes are explained in brief below.

### [Feedforward Attention based Text Classification Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_classifier/feedforward_attention_model)
The code is implemented based on the publication, [Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems](https://arxiv.org/abs/1512.08756). The paper proposes a simplified model of attention which is applicable to feed-forward neural networks and demonstrate that the resulting model can solve the synthetic addition and multiplication long-term memory problems for sequence lengths which are in terms to existing results.

Attention is an easier modeling of long-term dependencies. Attention mechanisms allow for a more direct dependence between the state of the model at different points in time. Following the definition from (Bahdanau et al., 2014), given a model which produces a hidden state at each time step, attention-based models compute a “context” vector ct as the weighted mean of the state sequence. The paper demonstrate that including an attention mechanism can allow a model to refer to specific points in a sequence when computing its output. They also provide an alternate argument for the claim made by Bahdanau et al. (2014) that attention helps models handle very long and widely variable-length sequences.

<img src="https://d3i71xaburhd42.cloudfront.net/87119572d1065fb079e1dee8fcdb6c4811143f96/250px/2-Figure1-1.png" width="360">

A limitation of the proposed model is that it will fail on any task where temporal order matters because computing an average over time discards order information. For example, on the two symbol temporal order task where a sequence must be classified in terms of whether two symbols X and Y appear in the order X, X; Y, Y ; X, Y ; or Y, X, our model can differentiate between the X, X and Y, Y cases perfectly but cannot differentiate between the X, Y and Y, X cases at all.

### [Parallel Recurrent based Text Classification Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_classifier/parallel_recurrent_model)
The code is implemented based on the publication, [Sliced Recurrent Neural Networks](https://arxiv.org/abs/1807.02291). Recurrent neural networks have achieved great success in many NLP tasks. However, they have difficulty in parallelization because of the recurrent structure, so it takes much time to train RNNs. The paper proposes sliced recurrent neural networks (SRNNs), which could be parallelized by slicing the sequences into many subsequences. SRNNs have the ability to obtain high-level information through multiple layers with few extra parameters.

<img src="https://d3i71xaburhd42.cloudfront.net/eefbe0d29fa9955caffc51777991cefbdbbaabab/250px/3-Figure2-1.png" width="360">

With the success of RNNs in many NLP tasks, many scholars have proposed different structures to increase the speed of RNNs. Most of the researches get faster by improving the recurrent units. However, the traditional connection structure has scarcely been questioned, in which each step is connected to its previous step. It is this connection structure that limits the speed of RNNs. The SRNN has improved the traditional connection method. Instead, a sliced structure is constructed to implement the parallelization of RNNs.

### [BERT based Text Classification Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_classifier/bert_model)
The code is implemented based on the publication, [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). The paper proposes a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1

<img src="https://pytorch.org/tutorials/_images/bert.png" width="360">

BERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.

BERT was trained with the masked language modeling (MLM) and next sentence prediction (NSP) objectives. It is efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation.

### [ALBERT based Text Classification Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_classifier/albert_model)
The code is implemented based on the publication, [ ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942). The paper proposes two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT:
a) Splitting the embedding matrix into two smaller matrices.
b) Using repeating layers split among groups.

Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations, longer training times, and unexpected model degradation. To address these problems, the paper proposes a two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows that the proposed methods lead to models that scale much better compared to the original BERT.
  
Paper also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, the ALBERT model establishes new state-of-the-art results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large.

<img src="https://pytorch.org/tutorials/_images/bert.png" width="360">

ALBERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left. It also uses repeating layers which results in a small memory footprint, however the computational cost remains similar to a BERT-like architecture with the same number of hidden layers as it has to iterate through the same number of (repeating) layers.

### [RoBERTa based Text Classification Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_classifier/roberta_model)
The code is implemented based on the publication, [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). 
RoBERTa is built on BERT and modified key hyperparameters and removes the next-sentence pretraining objective, and also training with much larger mini-batches and learning rates.

Language model pre-training has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, the paper shows that hyperparameter choices have significant impact on the final results. The paper shows that BERT was significantly undertrained, and can match or exceed the performance of every model published after it. RoBERTa models achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight the importance of previously overlooked design choices.

RoBERTa shows that performance can be substantially improved by training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data. 

<img src="https://pytorch.org/tutorials/_images/bert.png" width="360">

RoBERTa has the same architecture as BERT, but uses a byte-level BPE as a tokenizer (same as GPT-2) and uses a different pretraining scheme.

#### Reference
1. Zeping Yu, Gongshen Liu. 2018. "Sliced Recurrent Neural Networks". In Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018)
2. Colin Raffel, Daniel P. W. Ellis. 2016. "Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems". In Proceedings of the Neural and Evolutionary Computing (cs.NE)
3. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. 2019. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
4. Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. 2019. "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
5. https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html
6. https://huggingface.co/transformers/index.html
6. Git repository by zhedongzheng named tensorflow-nlp (https://github.com/zhedongzheng/tensorflow-nlp)