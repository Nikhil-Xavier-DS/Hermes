# [Enhanced Decomposable Attention Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_inference/enhanced_decomposable_attention_model)
The code is implemented based on the publication, [Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038). 
Reasoning and inference are central to human and artificial intelligence. Modeling inference in human language is very challenging. With the availability of large annotated data, it has recently become feasible to train neural network based inference models, which have shown to be very effective. The paper presents a new state-of-the-art model, achieving the accuracy of 88.6% on the Stanford Natural Language Inference Dataset. Unlike the previous top models that use very complicated network architectures, the proposed model first demonstrate that carefully designing sequential inference models based on chain LSTMs can outperform all previous models. 

<img src="https://raw.githubusercontent.com/coetaur0/ESIM/master/esim.png" width="360">

The paper also shows that by explicitly considering recursive architectures in both local inference modeling and inference composition, the model can achieve additional improvement. Particularly, incorporating syntactic parsing information, the model further improves the performance even when added to the already very strong model.

#### Reference
1. Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, Hui Jiang, Diana Inkpen. 2017. "Enhanced LSTM for Natural Language Inference". Proceedings of ACL 2017
2. https://nlp.stanford.edu/projects/snli
3. https://huggingface.co/transformers/index.html
4. https://www.pngitem.com/middle/TibbRTm_decomposable-attention-model-for-natural-language-inference-hd/
5. https://www.tensorflow.org
6. https://paperswithcode.com/paper/enhanced-lstm-for-natural-language-inference
