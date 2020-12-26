# [Parallel Recurrent based Text Classification Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_classifier/parallel_recurrent_model)
The code is implemented based on the publication, [Sliced Recurrent Neural Networks](https://arxiv.org/abs/1807.02291). Recurrent neural networks have achieved great success in many NLP tasks. However, they have difficulty in parallelization because of the recurrent structure, so it takes much time to train RNNs. The paper proposes sliced recurrent neural networks (SRNNs), which could be parallelized by slicing the sequences into many subsequences. SRNNs have the ability to obtain high-level information through multiple layers with few extra parameters.

<img src="https://d3i71xaburhd42.cloudfront.net/eefbe0d29fa9955caffc51777991cefbdbbaabab/250px/3-Figure2-1.png" width="360">

With the success of RNNs in many NLP tasks, many scholars have proposed different structures to increase the speed of RNNs. Most of the researches get faster by improving the recurrent units. However, the traditional connection structure has scarcely been questioned, in which each step is connected to its previous step. It is this connection structure that limits the speed of RNNs. The SRNN has improved the traditional connection method. Instead, a sliced structure is constructed to implement the parallelization of RNNs.

#### Reference
Zeping Yu, Gongshen Liu. 2018. Sliced Recurrent Neural Networks. In Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018)
