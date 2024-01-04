# FactorExe

> If factorization of computation is possible, very small models equipped with a search algorithm will perform arbitrarily complex computations.

## Code structure

- `bin_pack/`: old BinPack code, may be reused later.
- `c_vpr/`: code for the C-VPR task with `M` hops.
- `multiplication`: code for the `N*N` multiplication task.
- `regression/`: old quasi-linear regression code, maybe reused later.

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
- [Think before you speak: Training Language Models With Pause Tokens, [Goyal, 2023]](https://arxiv.org/abs/2310.02226)
- [Implicit Chain of Thought Reasoning via Knowledge Distillation, [Deng et al., 2023]](https://arxiv.org/abs/2311.01460)
- [Addressing Some Limitations of Transformers with Feedback Memory, [Fan et al., 2020]](https://arxiv.org/abs/2002.09402)
- [CoTFormer: More Tokens With Attention Make Up For Less Depth, [Mohtashami et al., 2023]](https://arxiv.org/abs/2310.10845)
- [Adaptive Computation with Elastic Input Sequence, [Xue et al., 2023]](https://arxiv.org/abs/2301.13195): generalization with respect to the computation sequence length.
- [The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers, [Csordás et al., 2021]](https://arxiv.org/abs/2108.12284)
- [Recurrent Independent Mechanisms, [Goyal et al., 2019]](https://arxiv.org/abs/1909.10893)
- [Transferring Inductive Biases through Knowledge Distillation, [Abnar et al., 2020]](https://arxiv.org/abs/2006.00555)
- [Pointer Value Retrieval: A new benchmark for understanding the limits of neural network generalization, [Zhang et al., 2021]](https://arxiv.org/abs/2107.12580): generalization, neural network reasonning, indirection.
- [Explaining grokking through circuit efficiency, [Varma et al., 2023]](https://arxiv.org/abs/2309.02390): on generalization versus memorization.
- [Thinking Like Transformers, [Weiss et al., 2021]](https://arxiv.org/abs/2106.06981): RASP language and computation model behind the transformer.
- [What Algorithms can Transformers Learn? A Study in Length Generalization, [Zhou et al., 2023]](https://arxiv.org/abs/2310.16028): on length generalization.
- [The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning, [Goldblum et al., 2023]](https://arxiv.org/abs/2304.05366): on no-free lunch theorem.
- [Lessons on Parameter Sharing across Layers in Transformers, [Takase & Kiyono, 2021]](https://arxiv.org/abs/2104.06022): superior performance when you stack layers multiple times (sharing params).
- [Understanding Parameter Sharing in Transformers, [Lin et al., 2023]](https://arxiv.org/abs/2306.09380)
- [Sparse Universal Transformer, [Tan et al., 2023]](https://arxiv.org/abs/2310.07096)
- [Improving the Neural GPU Architecture for Algorithm Learning, [Freivalds & Liepins, 2017]](https://arxiv.org/abs/1702.08727)
- [Neural GPUs Learn Algorithms, [Kaiser & Sutskever, 2015]](https://arxiv.org/abs/1511.08228)
