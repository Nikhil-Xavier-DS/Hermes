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

<img src="https://d3i71xaburhd42.cloudfront.net/eefbe0d29fa9955caffc51777991cefbdbbaabab/250px/3-Figure2-1.png" width="300">
