# [Match Pyramid Language Inference Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_inference/match_pyramid_model)
The code is implemented based on the publication, [Text Matching as Image Recognition](https://arxiv.org/abs/1602.06359). 
Matching two texts is a fundamental problem in many natural language processing tasks. An effective way is to extract meaningful matching patterns from words, phrases, and sentences to produce the matching score. Inspired by the success of convolutional neural network in image recognition, where neurons can capture many complicated patterns based on the extracted elementary visual patterns such as oriented edges and corners, the paper proposes to model text matching as the problem of image recognition. Firstly, a matching matrix whose entries represent the similarities between words is constructed and viewed as an image. Then a convolutional neural network is utilized to capture rich matching patterns in a layer-by-layer way.

<img src="https://www.mdpi.com/information/information-11-00421/article_deploy/html/images/information-11-00421-g007.png" width="360">

The model proposed can successfully identify salient signals such as n-gram and n-term matchings by resembling the compositional hierarchies of patterns in image recognition.

#### Reference
1. Liang Pang, Yanyan Lan, Jiafeng Guo, Jun Xu, Shengxian Wan, Xueqi Cheng. 2016. "Text Matching as Image Recognition"
2. https://nlp.stanford.edu/projects/snli
3. https://huggingface.co/transformers/index.html
4. https://www.tensorflow.org
