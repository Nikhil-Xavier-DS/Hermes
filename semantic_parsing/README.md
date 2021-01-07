# Semantic Parsing

Semantic parsing is the task of converting a natural language utterance to a logical form: a machine-understandable representation of its meaning. Semantic parsing can thus be understood as extracting the precise meaning of an utterance. Applications of semantic parsing include machine translation, question answering, ontology induction, automated reasoning, and code generation.

Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human languages, in particular, programming computers to process and analyze large amounts of natural language data. Deep neural network-style machine learning methods are widespread in natural language processing and can achieve state-of-the-art results in many natural language tasks. 

Some of the popular semantic parsing algorithms implemented in Hermes are explained in brief below.

### [Sequence to Sequence GRU Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/semantic_parsing/seq2seq_gru_model)
The code is implemented based on the publication, [Semantic Parsing for Task Oriented Dialog using Hierarchical Representations](https://arxiv.org/abs/1810.07942). 
Task oriented dialog systems typically first parse user utterances to semantic frames comprised of intents and slots. Previous work on task oriented intent and slot-filling work has been restricted to one intent per query and one slot label per token, and thus cannot model complex compositional requests. Alternative semantic parsing systems have represented queries as logical forms, but these are challenging to annotate and parse. The publication proposes a hierarchical annotation scheme for semantic parsing that allows the representation of compositional queries, and can be efficiently and accurately parsed by standard constituency parsing models.

<img src="https://d3i71xaburhd42.cloudfront.net/472a5227279b45f25508017816af34e3cb3ac0d7/2-Figure1-1.png" width="360">


#### Reference
1. Sonal Gupta, Rushin Shah, Mrinal Mohit, Anuj Kumar, Mike Lewis. 2018. "Semantic Parsing for Task Oriented Dialog using Hierarchical Representations". Conference on Empirical Methods in Natural Language Processing (EMNLP) 2018
2. https://github.com/facebookresearch/pytext/blob/master/pytext/docs/source/hierarchical_intent_slot_tutorial.rst
3. https://fb.me/semanticparsingdialog
4. Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, Hui Jiang, Diana Inkpen. 2017. "Enhanced LSTM for Natural Language Inference". Proceedings of ACL 2017
5. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. 2019. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
6. https://huggingface.co/transformers/index.html
7. Git repository: https://github.com/zhedongzheng/tensorflow-nlp
8. https://www.tensorflow.org
