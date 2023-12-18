# FactorExe

> If factorisation of computation is possible, very small models equipped with a search algorithm will be able to perform arbitrarily complex computations.

## Questions

- Should intermediate computations be discretely sampled or continuous (e.g. full embedding). If the former, one will need RL to train the system. If the latter, then one can backpropagate through the chain of thought and it looks more like an RNN (with attention layers if one uses, say, a transformer). You can think of the latter as a full transformer with weight sharing between layers.

- The above question triggers this one: can a deep model with lots of layers be factorised into a small model repeated many times? This links to adaptive compute, universal transformer, mixture of experts, etc.

- What is the best algorithm for exploiting a factorised model? A deep transformer could be seen as doing chain-of-thought layer by layer. Doing tree search on top of a factorised model would lead to no equivalent in the NN architecture domain.

- Can the results of these intermediate computations be discovered (i.e. only using supervision from the final output) efficiently?

## Initial POC

- Study the scaling laws of factorised computation. How does performance on a simple computational task evolve as a function of model size, chain of thought length, and training data?
