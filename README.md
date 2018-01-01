--- Created by Peter Cai ----

[Venusodeday]
A repository to store all the frequent-used algorithms for researches or personal uses.

1. BP neural network
Back-propagation neural network is developed from mullti-layer perceptron (MLP). In order to extend the MLP to nonlinear mapping, it uses activation funtions (Sigmoid, tanh, etc.) to transform the output of neural units. Inspired by gradient descent method, the errors (usually referring to mean square errors between real value and model output) can be propagated back to adjust the weights which connected adjcent layers.

The program considers the idea of object-oriented programming. It defined some classes represent neural units, layers and net. All the units  need to get inputs, weights connected to last layer, and activation fucntions. Then it consist of some methods, including output calculated method, error calculated method and weights updata methods, etc.

A regular BP neural network can approximate any nolinear functions, whereas it doesn't have good convengenercy with few iteration steps.