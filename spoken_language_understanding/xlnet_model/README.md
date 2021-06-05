# [XLNet based Spoken Language Understanding Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/spoken_language_understanding/xlnet_model)
The code is implemented based on the publication, [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237). 
With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling.  However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons, XLNet model is proposed. XLNet is a generalized autoregressive pretraining method that enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and overcoming the limitations of BERT model due to autoregressive formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining. XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking.

<img src="https://i0.wp.com/mlexplained.com/wp-content/uploads/2019/06/Screen-Shot-2019-06-22-at-5.38.12-PM.png?resize=1024%2C567&ssl=1" width="360">

Due to the difficulty of training a fully auto-regressive model over various factorization order, XLNet is pretrained using only a sub-set of the output tokens as target which are selected. XLNet is one of the few models that has no sequence length limit.


#### Reference
1. Xiaodong Zhang and Houfeng Wang. 2016. "A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding". Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence (IJCAI-16)
2. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. 2019. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
3. Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. 2019. "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
4. Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le. 2020. "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
5. https://www.tensorflow.org
6. https://huggingface.co/transformers/index.html