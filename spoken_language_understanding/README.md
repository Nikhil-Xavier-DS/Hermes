# Spoken Language Understanding (Intent Detection and Slot Filling)

Spoken language understanding (SLU) is an emerging field in between the areas of speech processing and natural language processing. The term spoken language understanding has largely been coined for targeted understanding of human speech directed at machines. For natural language inference, the algorithms use the ATIS dataset.

The ATIS dataset is a standard benchmark dataset widely used as an intent classification. ATIS Stands for Airline Travel Information System. Intent classification is an important component of Natural Language Understanding (NLU) systems in any chatbot platform.

Some of the popular spoken language understanding algorithms implemented in Hermes are explained in brief below.

### [Bidirectional LSTM based Spoken Language Understanding Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/spoken_language_understanding/bi_lstm_model)
The code is implemented based on the publication, [A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding](https://www.ijcai.org/Proceedings/16/Papers/425.pdf). 
Two major tasks in spoken language understanding (SLU) are intent determination (ID) and slot filling (SF). Recurrent neural networks (RNNs) have been proved effective in SF, while there is no prior work using RNNs in ID. Based on the idea that the intent and semantic slots of a sentence are correlative, a joint model for both tasks. Gated recurrent unit (GRU) is used to learn the representation of each time step, by which the label of each slot is predicted. Meanwhile, a max-pooling layer is employed to capture global features of a sentence for intent classification. The representations are shared by two tasks and the model is trained by a united loss function.

<img src="https://d3i71xaburhd42.cloudfront.net/1f9e2d6df1eaaf04aebf428d9fa9a9ffc89e373c/3-Figure1-1.png" width="360">

The model uses LSTM instead of GRU as proposed in the publication.

### [Bidirectional GRU based Spoken Language Understanding Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/spoken_language_understanding/bi_lstm_model)
The code is implemented based on the publication, [A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding](https://www.ijcai.org/Proceedings/16/Papers/425.pdf). 
Two major tasks in spoken language understanding (SLU) are intent determination (ID) and slot filling (SF). Recurrent neural networks (RNNs) have been proved effective in SF, while there is no prior work using RNNs in ID. Based on the idea that the intent and semantic slots of a sentence are correlative, a joint model for both tasks. Gated recurrent unit (GRU) is used to learn the representation of each time step, by which the label of each slot is predicted. Meanwhile, a max-pooling layer is employed to capture global features of a sentence for intent classification. The representations are shared by two tasks and the model is trained by a united loss function.

<img src="https://d3i71xaburhd42.cloudfront.net/1f9e2d6df1eaaf04aebf428d9fa9a9ffc89e373c/3-Figure1-1.png" width="360">

#### Reference
1. Liang Pang, Yanyan Lan, Jiafeng Guo, Jun Xu, Shengxian Wan, Xueqi Cheng. 2016. "Text Matching as Image Recognition"
2. Ankur P. Parikh, Oscar Täckström, Dipanjan Das, Jakob Uszkoreit. 2016. "A Decomposable Attention Model for Natural Language Inference". Proceeedings of EMNLP 2016
3. https://www.microsoft.com/en-us/research/project/spoken-language-understanding/
4. https://huggingface.co/transformers/index.html
5. Git repository: https://github.com/zhedongzheng/tensorflow-nlp
6. https://www.tensorflow.org
7. https://github.com/yvchen/JointSLU/tree/master/data


