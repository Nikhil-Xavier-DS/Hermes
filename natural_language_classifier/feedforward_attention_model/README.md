# [Feedforward Attention based Text Classification Models](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/natural_language_classifier/feedforward_attention_model)
The code is implemented based on the publication, [Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems](https://arxiv.org/abs/1512.08756). The paper proposes a simplified model of attention which is applicable to feed-forward neural networks and demonstrate that the resulting model can solve the synthetic addition and multiplication long-term memory problems for sequence lengths which are in terms to existing results.

Attention is an easier modeling of long-term dependencies. Attention mechanisms allow for a more direct dependence between the state of the model at different points in time. Following the definition from (Bahdanau et al., 2014), given a model which produces a hidden state at each time step, attention-based models compute a “context” vector ct as the weighted mean of the state sequence. The paper demonstrate that including an attention mechanism can allow a model to refer to specific points in a sequence when computing its output. They also provide an alternate argument for the claim made by Bahdanau et al. (2014) that attention helps models handle very long and widely variable-length sequences.

<img src="https://d3i71xaburhd42.cloudfront.net/87119572d1065fb079e1dee8fcdb6c4811143f96/250px/2-Figure1-1.png" width="360">

A limitation of the proposed model is that it will fail on any task where temporal order matters because computing an average over time discards order information. For example, on the two symbol temporal order task where a sequence must be classified in terms of whether two symbols X and Y appear in the order X, X; Y, Y ; X, Y ; or Y, X, our model can differentiate between the X, X and Y, Y cases perfectly but cannot differentiate between the X, Y and Y, X cases at all.

#### Reference
Colin Raffel, Daniel P. W. Ellis. 2016. Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems. In Proceedings of the Neural and Evolutionary Computing (cs.NE)
