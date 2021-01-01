# [Decomposable Attention based Natural Language Inference Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_inference/decomposable_attention_model)
The code is implemented based on the publication, [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933). 
The paper proposes a simple neural architecture for natural language inference. The approach proposed uses attention to decompose the problem into sub-problems that can be solved separately, thus making it trivially parallelizable. On the Stanford Natural Language Inference (SNLI) dataset, the decomposable attention model churns out state-of-the-art results with almost an order of magnitude fewer parameters than previous work and without relying on any word-order information. Adding intra-sentence attention that takes a minimum amount of order into account yields further improvements.

<img src="https://www.pngitem.com/pimgs/m/578-5787448_decomposable-attention-model-for-natural-language-inference-hd.png" width="360">

#### Reference
1. Ankur P. Parikh, Oscar Täckström, Dipanjan Das, Jakob Uszkoreit. 2016. "A Decomposable Attention Model for Natural Language Inference". Proceeedings of EMNLP 2016
2. https://nlp.stanford.edu/projects/snli
3. https://huggingface.co/transformers/index.html
4. https://www.pngitem.com/middle/TibbRTm_decomposable-attention-model-for-natural-language-inference-hd/
4. https://www.tensorflow.org