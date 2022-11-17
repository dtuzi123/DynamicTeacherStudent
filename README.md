
# Dynamic Self-Supervised Teacher-Student Network Learning

>📋 This is the implementation of Dynamic Self-Supervised Teacher-Student Network Learning


# Title : Dynamic Self-Supervised Teacher-Student Network Learning

# Paper link : 



# Abstract

Lifelong learning (LLL) represents the ability of an artificial intelligence system to learn successively a sequence of different databases.  In this paper we introduce the Dynamic Self-Supervised Teacher-Student Network (D-TS), representing a more general LLL framework, where the Teacher is implemented as a dynamically expanding mixture model which automatically increases its capacity to deal with a growing number of tasks. We propose the Knowledge Discrepancy Score (KDS) criterion for measuring the relevance of the incoming information characterizing a new task when compared to the existing knowledge accumulated by the Teacher module from its previous training. The KDS ensures a light Teacher architecture while also enabling to reuse the learned knowledge whenever appropriate, accelerating the learning of given tasks. The Student module is implemented as a lightweight probabilistic generative model. We introduce a novel self-supervised learning for the Student that allows to capture cross-domain latent representations from the entire knowledge accumulated by the Teacher as well as from novel data. We perform several experiments which show that D-TS can achieve the state of the art results in LLL while requiring fewer parameters than other methods.

# Environment

1. Tensorflow 2.1
2. Python 3.6

# Training and evaluation

>📋 Python xxx.py, the model will be automatically trained and then report the results after the training.

>📋 Different parameter settings of D-TS would lead different results and we also provide different settings used in our experiments.

# BibTex
>📋 If you use our code, please cite our paper as:



