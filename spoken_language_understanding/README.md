# Spoken Language Understanding (Intent Detection and Slot Filling)

Spoken language understanding (SLU) is an emerging field in between the areas of speech processing and natural language processing. The term spoken language understanding has largely been coined for targeted understanding of human speech directed at machines. For natural language inference, the algorithms use the ATIS dataset.

The ATIS dataset is a standard benchmark dataset widely used as an intent classification. ATIS Stands for Airline Travel Information System. Intent classification is an important component of Natural Language Understanding (NLU) systems in any chatbot platform.

Some of the popular spoken language understanding algorithms implemented in Hermes are explained in brief below.

### [Bidirectional LSTM based Spoken Language Understanding Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/spoken_language_understanding/bi_lstm_model)
The code is implemented based on the publication, [A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding](https://www.ijcai.org/Proceedings/16/Papers/425.pdf). 
Two major tasks in spoken language understanding (SLU) are intent determination (ID) and slot filling (SF). Recurrent neural networks (RNNs) have been proved effective in SF, while there is no prior work using RNNs in ID. Based on the idea that the intent and semantic slots of a sentence are correlative, a joint model for both tasks. Gated recurrent unit (GRU) is used to learn the representation of each time step, by which the label of each slot is predicted. Meanwhile, a max-pooling layer is employed to capture global features of a sentence for intent classification. The representations are shared by two tasks and the model is trained by a united loss function.

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQmNX4AEt-aMIfkp4OsEm7SzYB-fzPf62dexg&usqp=CAU" width="360">

The model uses LSTM instead of GRU as proposed in the publication.

### [Transformer based Spoken Language Understanding Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/spoken_language_understanding/transformer_model)
The code is implemented based on the publication, [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. The publication proposes a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. 

<img src="https://hub.packtpub.com/wp-content/uploads/2018/04/Attention.png" width="360">

### [Bidirectional GRU based Spoken Language Understanding Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/spoken_language_understanding/bi_lstm_model)
The code is implemented based on the publication, [A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding](https://www.ijcai.org/Proceedings/16/Papers/425.pdf). 
Two major tasks in spoken language understanding (SLU) are intent determination (ID) and slot filling (SF). Recurrent neural networks (RNNs) have been proved effective in SF, while there is no prior work using RNNs in ID. Based on the idea that the intent and semantic slots of a sentence are correlative, a joint model for both tasks. Gated recurrent unit (GRU) is used to learn the representation of each time step, by which the label of each slot is predicted. Meanwhile, a max-pooling layer is employed to capture global features of a sentence for intent classification. The representations are shared by two tasks and the model is trained by a united loss function.

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQmNX4AEt-aMIfkp4OsEm7SzYB-fzPf62dexg&usqp=CAU" width="360">

### [BERT based Spoken Language Understanding Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/spoken_language_understanding/bert_model)
The code is implemented based on the publication, [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). The paper proposes a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1

BERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.

BERT was trained with the masked language modeling (MLM) and next sentence prediction (NSP) objectives. It is efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation.

###[ALBERT based Spoken Language Understanding Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/spoken_language_understanding/albert_model)
The code is implemented based on the publication, [ ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942). The paper proposes two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT:
a) Splitting the embedding matrix into two smaller matrices.
b) Using repeating layers split among groups.

Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations, longer training times, and unexpected model degradation. To address these problems, the paper proposes a two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows that the proposed methods lead to models that scale much better compared to the original BERT.
  
Paper also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, the ALBERT model establishes new state-of-the-art results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large.

<img src="https://pytorch.org/tutorials/_images/bert.png" width="360">

ALBERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left. It also uses repeating layers which results in a small memory footprint, however the computational cost remains similar to a BERT-like architecture with the same number of hidden layers as it has to iterate through the same number of (repeating) layers.

### [XLNet based Spoken Language Understanding Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/spoken_language_understanding/xlnet_model)
The code is implemented based on the publication, [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237). 
With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling.  However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons, XLNet model is proposed. XLNet is a generalized autoregressive pretraining method that enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and overcoming the limitations of BERT model due to autoregressive formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining. XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking.

<img src="https://i0.wp.com/mlexplained.com/wp-content/uploads/2019/06/Screen-Shot-2019-06-22-at-5.38.12-PM.png?resize=1024%2C567&ssl=1" width="360">

Due to the difficulty of training a fully auto-regressive model over various factorization order, XLNet is pretrained using only a sub-set of the output tokens as target which are selected. XLNet is one of the few models that has no sequence length limit.

### [RoBERTa based Spoken Language Understanding Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/spoken_language_understanding/roberta_model)
The code is implemented based on the publication, [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). 
RoBERTa is built on BERT and modified key hyperparameters and removes the next-sentence pretraining objective, and also training with much larger mini-batches and learning rates.

Language model pre-training has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, the paper shows that hyperparameter choices have significant impact on the final results. The paper shows that BERT was significantly undertrained, and can match or exceed the performance of every model published after it. RoBERTa models achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight the importance of previously overlooked design choices.

RoBERTa shows that performance can be substantially improved by training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data. 

<img src="https://pytorch.org/tutorials/_images/bert.png" width="360">

RoBERTa has the same architecture as BERT, but uses a byte-level BPE as a tokenizer (same as GPT-2) and uses a different pretraining scheme.


#### Reference
1. Xiaodong Zhang and Houfeng Wang. 2016. "A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding". Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence (IJCAI-16)
2. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. 2017. "Attention Is All You Need". Proceeedings of EMNLP 2016
3. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. 2019. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
4. Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. 2019. "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
5. Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le. 2020. "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
6. https://www.microsoft.com/en-us/research/project/spoken-language-understanding/
7. Git repository: https://github.com/zhedongzheng/tensorflow-nlp
8. https://www.tensorflow.org
9. https://github.com/yvchen/JointSLU/tree/master/data
10. https://huggingface.co/transformers/index.html


