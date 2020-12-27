# Natural Language Inference
Natural language inference is the task of determining whether a “hypothesis” is true (entailment), false (contradiction), or undetermined (neutral) given a “premise”. For natural language inference, the algorithms use Standard Natural Language Inference (SNLI) dataset.

The Stanford Natural Language Inference (SNLI) Corpus contains around 550k hypothesis/premise pairs. Models are evaluated based on accuracy. The SNLI corpus (version 1.0) is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference (NLI), also known as recognizing textual entailment (RTE). The corpus serves both as a benchmark for evaluating representational systems for text, especially including those induced by representation learning methods, as well as a resource for developing NLP models of any kind.

Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human languages, in particular, programming computers to process and analyze large amounts of natural language data. Deep neural network-style machine learning methods are widespread in natural language processing and can achieve state-of-the-art results in many natural language tasks. 

Some of the popular natural language inference algorithms implemented in Hermes are explained in brief below.

### [Decomposable Attention based Natural Language Inference Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_inference/decomposable_attention_model)
The code is implemented based on the publication, [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933). 
The paper proposes a simple neural architecture for natural language inference. The approach proposed uses attention to decompose the problem into sub-problems that can be solved separately, thus making it trivially parallelizable. On the Stanford Natural Language Inference (SNLI) dataset, the decomposable attention model churns out state-of-the-art results with almost an order of magnitude fewer parameters than previous work and without relying on any word-order information. Adding intra-sentence attention that takes a minimum amount of order into account yields further improvements.

<img src="https://www.pngitem.com/pimgs/m/578-5787448_decomposable-attention-model-for-natural-language-inference-hd.png" width="360">

### [Match Pyramid Language Inference Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_inference/match_pyramid_model)
The code is implemented based on the publication, [Text Matching as Image Recognition](https://arxiv.org/abs/1602.06359). 
Matching two texts is a fundamental problem in many natural language processing tasks. An effective way is to extract meaningful matching patterns from words, phrases, and sentences to produce the matching score. Inspired by the success of convolutional neural network in image recognition, where neurons can capture many complicated patterns based on the extracted elementary visual patterns such as oriented edges and corners, the paper proposes to model text matching as the problem of image recognition. Firstly, a matching matrix whose entries represent the similarities between words is constructed and viewed as an image. Then a convolutional neural network is utilized to capture rich matching patterns in a layer-by-layer way.

<img src="https://www.mdpi.com/information/information-11-00421/article_deploy/html/images/information-11-00421-g007.png" width="360">

The model proposed can successfully identify salient signals such as n-gram and n-term matchings by resembling the compositional hierarchies of patterns in image recognition.

### [Enhanced Sequential Inference Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_inference/enhanced_sequential_inference_model)
The code is implemented based on the publication, [Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038). 
Reasoning and inference are central to human and artificial intelligence. Modeling inference in human language is very challenging. With the availability of large annotated data, it has recently become feasible to train neural network based inference models, which have shown to be very effective. The paper presents a new state-of-the-art model, achieving the accuracy of 88.6% on the Stanford Natural Language Inference Dataset. Unlike the previous top models that use very complicated network architectures, the proposed model first demonstrate that carefully designing sequential inference models based on chain LSTMs can outperform all previous models. 

<img src="https://raw.githubusercontent.com/coetaur0/ESIM/master/esim.png" width="360">

The paper also shows that by explicitly considering recursive architectures in both local inference modeling and inference composition, the model can achieve additional improvement. Particularly, incorporating syntactic parsing information, the model further improves the performance even when added to the already very strong model.

### [BERT based Natural Language Inference Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_inference/bert_model)
The code is implemented based on the publication, [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). The paper proposes a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1

<img src="https://pytorch.org/tutorials/_images/bert.png" width="360">

BERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.

BERT was trained with the masked language modeling (MLM) and next sentence prediction (NSP) objectives. It is efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation.

###[ALBERT based Natural Language Inference Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_inference/albert_model)
The code is implemented based on the publication, [ ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942). The paper proposes two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT:
a) Splitting the embedding matrix into two smaller matrices.
b) Using repeating layers split among groups.

Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations, longer training times, and unexpected model degradation. To address these problems, the paper proposes a two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows that the proposed methods lead to models that scale much better compared to the original BERT.
  
Paper also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, the ALBERT model establishes new state-of-the-art results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large.

<img src="https://pytorch.org/tutorials/_images/bert.png" width="360">

ALBERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left. It also uses repeating layers which results in a small memory footprint, however the computational cost remains similar to a BERT-like architecture with the same number of hidden layers as it has to iterate through the same number of (repeating) layers.

### [RoBERTa based Natural Language Inference Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_inference/roberta_model)
The code is implemented based on the publication, [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). 
RoBERTa is built on BERT and modified key hyperparameters and removes the next-sentence pretraining objective, and also training with much larger mini-batches and learning rates.

Language model pre-training has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, the paper shows that hyperparameter choices have significant impact on the final results. The paper shows that BERT was significantly undertrained, and can match or exceed the performance of every model published after it. RoBERTa models achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight the importance of previously overlooked design choices.

RoBERTa shows that performance can be substantially improved by training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data. 

<img src="https://pytorch.org/tutorials/_images/bert.png" width="360">

RoBERTa has the same architecture as BERT, but uses a byte-level BPE as a tokenizer (same as GPT-2) and uses a different pretraining scheme.

### [XLNet based Natural Language Inference Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_inference/xlnet_model)
The code is implemented based on the publication, [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237). 
With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling.  However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons, XLNet model is proposed. XLNet is a generalized autoregressive pretraining method that enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and overcoming the limitations of BERT model due to autoregressive formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining. XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking.

<img src="https://i0.wp.com/mlexplained.com/wp-content/uploads/2019/06/Screen-Shot-2019-06-22-at-5.38.12-PM.png?resize=1024%2C567&ssl=1" width="360">

Due to the difficulty of training a fully auto-regressive model over various factorization order, XLNet is pretrained using only a sub-set of the output tokens as target which are selected. XLNet is one of the few models that has no sequence length limit.

#### Reference
1. Liang Pang, Yanyan Lan, Jiafeng Guo, Jun Xu, Shengxian Wan, Xueqi Cheng. 2016. "Text Matching as Image Recognition"
2. Ankur P. Parikh, Oscar Täckström, Dipanjan Das, Jakob Uszkoreit. 2016. "A Decomposable Attention Model for Natural Language Inference". Proceeedings of EMNLP 2016
3. Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, Hui Jiang, Diana Inkpen. 2017. "Enhanced LSTM for Natural Language Inference". Proceedings of ACL 2017
4. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. 2019. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
5. Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. 2019. "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
6. Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le. 2020. "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
7. https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html
8. https://huggingface.co/transformers/index.html
9. Git repository: https://github.com/zhedongzheng/tensorflow-nlp
10. https://www.tensorflow.org
11. https://mlexplained.com/2019/06/30/paper-dissected-xlnet-generalized-autoregressive-pretraining-for-language-understanding-explained
12. https://www.pngitem.com/middle/TibbRTm_decomposable-attention-model-for-natural-language-inference-hd/
13. https://nlp.stanford.edu/projects/snli
14. https://paperswithcode.com/paper/enhanced-lstm-for-natural-language-inference


