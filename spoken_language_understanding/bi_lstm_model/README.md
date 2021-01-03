# [Bidirectional LSTM based Spoken Language Understanding Model](https://github.com/Nikhil-Xavier-DS/Hermes/tree/master/spoken_language_understanding/bi_lstm_model)
The code is implemented based on the publication, [A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding](https://www.ijcai.org/Proceedings/16/Papers/425.pdf). 
Two major tasks in spoken language understanding (SLU) are intent determination (ID) and slot filling (SF). Recurrent neural networks (RNNs) have been proved effective in SF, while there is no prior work using RNNs in ID. Based on the idea that the intent and semantic slots of a sentence are correlative, a joint model for both tasks. Gated recurrent unit (GRU) is used to learn the representation of each time step, by which the label of each slot is predicted. Meanwhile, a max-pooling layer is employed to capture global features of a sentence for intent classification. The representations are shared by two tasks and the model is trained by a united loss function.

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQmNX4AEt-aMIfkp4OsEm7SzYB-fzPf62dexg&usqp=CAU" width="360">

The model uses LSTM instead of GRU as proposed in the publication.

#### Reference
1. Xiaodong Zhang and Houfeng Wang. 2016. "A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding". Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence (IJCAI-16)
2. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. 2017. "Attention Is All You Need". Proceeedings of EMNLP 2016
3. https://www.microsoft.com/en-us/research/project/spoken-language-understanding/
5. Git repository: https://github.com/zhedongzheng/tensorflow-nlp
6. https://www.tensorflow.org
7. https://github.com/yvchen/JointSLU/tree/master/data