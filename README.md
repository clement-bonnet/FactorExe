# FactorExe

> If factorization of computation is possible, very small models equipped with a search algorithm will perform arbitrarily complex computations.

## Questions

- Should intermediate computations be discretely sampled or continuous (e.g. full embedding)? If the former, one will need RL to train the system. If the latter, one can backpropagate through the chain of thought which looks more like an RNN (with attention layers if one uses a small transformer as the factorized model). It is equivalent to a full transformer with complete weight sharing between layers.

- The above question triggers this one: can a deep model with lots of layers be factorized into a small model repeated many times? This links to adaptive compute, universal transformer, mixture of experts, etc. The number of repetitions of the small model may have to be higher than the ratio deep_model_size/small_model_size due to factorization.

- What is the best algorithm for exploiting a factorized model? A deep transformer could be seen as doing chain of thought layer by layer. Doing a tree search on top of a factorized model would probably increase its abilities and does not have any equivalent in the NN architecture domain.

- Can the results of these intermediate computations be discovered (i.e. only using supervision from the final output) efficiently? Chain of thought works well because the chains are supervised, i.e. they exist in the training data. Can we discover these chains from unsupervised, potentially reinforcement-learned data?

## Potential of Factorized Computation

We could end up with very small LLMs matching or outperforming the largest ones (e.g. GPT-4), which would be incredibly sick and revolutionary. Also, it would be possible to query these factorized models for arbitrary compute budgets, giving rise to a controllable compute-performance trade-off. Technically, one could even imagine a manageable compute-safety trade-off by searching the computation tree for "safe" states given by a value.

## Initial POC

- Study the scaling laws of factorized computation. How does performance on a simple computational task evolve as a function of model size, chain of thought length, and training data? Can one recover similar performance by factorizing a model? Does one get higher sample efficiency assuming the chains are given? Does chain-of-thought factorization lead to linear, sublinear or superlinear scaling when increasing the chain length?
