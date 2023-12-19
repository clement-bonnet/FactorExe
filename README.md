# FactorExe

> If factorization of computation is possible, very small models equipped with a search algorithm will perform arbitrarily complex computations.

## Questions

- Should intermediate computations be discretely sampled or continuous (e.g. full embedding)? If the former, one will need RL to train the system. If the latter, one can backpropagate through the chain of thought which looks more like an RNN (with attention layers if one uses a small transformer as the factorized model). It is equivalent to a full transformer with complete weight sharing between layers.

- The above question triggers this one: can a deep model with lots of layers be factorized into a small model repeated many times? This links to adaptive compute, universal transformer, mixture of experts, etc. The number of repetitions of the small model may have to be higher than the ratio deep_model_size/small_model_size due to factorization.

- What is the best algorithm for exploiting a factorized model? A deep transformer could be seen as doing chain of thought layer by layer. Doing a tree search on top of a factorized model would probably increase its abilities and does not have any equivalent in the NN architecture domain.

- Can the results of these intermediate computations be discovered (i.e. only using supervision from the final output) efficiently? Chain of thought works well because the chains are supervised, i.e. they exist in the training data. Can we discover these chains from unsupervised, potentially reinforcement-learned data?

## Potential of Factorized Computation

We could end up with very small LLMs matching or outperforming the largest ones (e.g. GPT-4), which would be incredibly sick and revolutionary. Also, it would be possible to query these factorized models for arbitrary compute budgets, giving rise to a controllable compute-performance trade-off.

On a tangent, two other applications could emerge. First, one could even imagine a manageable compute-safety trade-off by searching the computation tree for "safe" states given by a value (see LeCun's JEPA). Second, if the search heuristic is well calibrated, one could imagine estimating uncertainty during the search, potentially leading to automatic search budgets.

## Initial POC

1. Study the scaling laws of factorized computation.
    1. How does performance on a simple computational task (e.g. arithmetic) evolve as a function of model size, chain of thought length, and training data?
    2. Can one recover similar performance by factorizing a model?
    3. Does one get higher sample efficiency than 1-shot models assuming the chains are given?
    4. Does chain-of-thought factorization lead to linear, sublinear or superlinear scaling when increasing the chain length?

## Resources

- [Teaching Arithmetic to Small Transformers, [Lee et al., 2023]](https://arxiv.org/abs/2307.03381)

- [Universal Transformers, [Dehghani et al., 2019]](https://arxiv.org/abs/1807.03819)

- [Adaptivity and Modularity for Efficient Generalization Over Task Complexity, [Abnar et al., 2023]](https://arxiv.org/abs/2310.08866)

- [Adaptive Computation Time for Recurrent Neural Networks, [Graves, 2017]](https://arxiv.org/abs/1603.08983)

- [Modular Deep Learning, [Pfeiffer et al., 2023]](https://arxiv.org/abs/2302.11529)

- [Adaptive Computation Time for Recurrent Neural Networks, [Graves, 2016]](https://arxiv.org/abs/1603.08983)

- [PonderNet: Learning to Ponder, [Banino et al., 2021]](https://arxiv.org/abs/2107.05407)

- [An investigation of model-free planning, [Guez et al., 2019]](https://arxiv.org/abs/1901.03559)

- [Chain of Code: Reasoning with a Language Model-Augmented Code Emulator, [Li et al., 2023]](https://arxiv.org/abs/2312.04474)
