# Official course map - learn-deep-learning-with-phoebe

**Course:** Deep Learning Foundations - neural nets from the ground up: backprop intuition, PyTorch, and when deep learning beats trees.
**Arc:** single-track builder 6 sessions, Karpathy build-it-yourself ethos. Running artifact: one real neural network - built as scalars on paper (s2), driven live in the browser (s3), rebuilt in PyTorch (s3-s5), and honestly benchmarked against the tree course's booster on identical data (s6).
**Bucket:** `ds` d4 builder project (hub entry exists as status:planned - FLIP, do not append). Palette: deep-water cyan #0E7490 ramp + amber #F59E0B (recommendation owns blue #2563EB, finance owns green-teal #0F766E - this is the cyan lane).
**Seams:** `learn-ensemble-methods` owns trees; s6 imports its 73.0 benchmark on byte-identical data. `learn-intro-ml` owns the ML lifecycle. `learn-ai-*` GenAI shelf owns transformers/LLMs - this course stops at embeddings + the modality map and points there. Parked `learn-ml-epistemology` owns the deep double-descent/generalization philosophy; s5 carries a one-card teaser only.
**Build mode:** course-taking loop PAUSED - built direct from verified sources.

## Source universe (verified 2026-09-03 by research agent; AlexNet quotes extracted from the proceedings PDF itself)

| # | Source | Citation + URL | Usable claim | Honesty caveat (MUST carry) | Maps to |
|---|--------|----------------|--------------|------------------------------|---------|
| 1 | Rumelhart, Hinton & Williams 1986 | "Learning representations by back-propagating errors", Nature 323. https://www.nature.com/articles/323533a0 | Backprop "repeatedly adjusts the weights of connections in the network so as to minimize" output error; hidden units "come to represent important features of the task domain". The pitch is LEARNED INTERNAL REPRESENTATIONS. | "Backprop invented 1986" is FALSE: Linnainmaa 1970 (reverse-mode AD), Werbos 1974/1982 (applied to networks); lineage citable via Schmidhuber 2015 (arXiv:1404.7828, sec 5.5). 1986 = when it was shown to work and matter. | s2 |
| 2 | Universal approximation | Cybenko 1989, Math. Control Signals Systems 2 (https://link.springer.com/article/10.1007/BF02551274) + Hornik, Stinchcombe & White 1989, Neural Networks 2(5) | One hidden layer + sigmoidal/squashing nonlinearity approximates any continuous function on compact sets ("arbitrary decision regions can be arbitrarily well approximated... with only a single internal, hidden layer"), "provided sufficiently many hidden units are available". | EXISTENCE ONLY - says nothing about whether gradient descent finds it, sample count, or width (can be exponential). "UAT means NNs can learn anything" is the classic misread. The spiral demo IS this caveat made visible. | s1, s3 |
| 3 | LeCun et al. 1998 LeNet-5 | "Gradient-Based Learning Applied to Document Recognition", Proc. IEEE 86(11). https://leon.bottou.org/papers/lecun-98h | Convolution = inductive bias: local receptive fields, shared weights, spatial subsampling - shift invariance + drastic parameter cuts. LeNet-5 MNIST test error 0.95% (0.8% with distortion augmentation). | 0.7% belongs to boosted LeNet-4 (ensemble), not LeNet-5. CNN work dates to LeCun 1989 zip-code nets; credit Fukushima's Neocognitron 1980 for the lineage. | s4 |
| 4 | Srivastava et al. 2014 dropout | JMLR 15. https://www.jmlr.org/papers/v15/srivastava14a.html | Verbatim: "The key idea is to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much." SOTA improvements across vision/speech/text/biology at the time. | Approximates averaging exponentially many thinned nets - an approximation argument, not a literal ensemble. Modern use: mostly FC heads + transformers, less in conv layers (norm-layer era). | s5 |
| 5 | Adam + AdamW | Kingma & Ba, ICLR 2015 (arXiv:1412.6980) + Loshchilov & Hutter, ICLR 2019 (arXiv:1711.05101) | Adam: "first-order gradient-based optimization... based on adaptive estimates of lower-order moments", suited to noisy/sparse gradients. AdamW verbatim: L2 and weight decay "are equivalent for standard stochastic gradient descent... but as we demonstrate this is not the case for adaptive gradient algorithms, such as Adam"; decoupling "substantially improves Adam's generalization". | Cite Adam as ICLR 2015. AdamW (2019) is TODAY'S default (torch.optim.AdamW). Never claim Adam always beats SGD - tuned SGD+momentum stayed competitive on vision. | s3, s5 |
| 6 | Karpathy corpus | micrograd https://github.com/karpathy/micrograd + nn-zero-to-hero https://github.com/karpathy/nn-zero-to-hero + "A Recipe for Training Neural Networks" (2019) https://karpathy.github.io/2019/04/25/recipe/ | micrograd README verbatim: "A tiny Autograd engine... Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG" - engine ~100 lines, scalar-valued. Recipe rules verbatim: "neural net training fails silently"; "Become one with the data"; "Verify loss @ init"; "Overfit one batch... as little as two"; "Don't be a hero"; "get more data - the by far best and preferred way to regularize a model in any practical setting". | Recipe's six-step order: data → skeleton + dumb baselines → overfit → regularize → tune → squeeze. | s2, s3, s5 |
| 7 | Double descent | Belkin et al., PNAS 116(32) 2019 (arXiv:1812.11118) + Nakkiran et al. 2019 (arXiv:1912.02292) | Belkin verbatim: "This 'double descent' curve subsumes the textbook U-shaped bias-variance trade-off curve by showing how increasing model capacity beyond the point of interpolation results in improved performance." Nakkiran: occurs in modern nets model-wise AND epoch-wise; regimes where more data hurts. | Most visible with label noise near the interpolation threshold; does NOT mean bigger-is-always-better or regularization obsolete. One teaser card only - the parked epistemology course owns the full story. | s5 |
| 8 | Zhang et al. rethinking generalization | ICLR 2017 (arXiv:1611.03530); CACM 2021 republication | Nets "easily fit a random labeling of the training data"; unchanged by explicit regularization; classical complexity measures cannot explain generalization. | Shows classical tools INSUFFICIENT, not that generalization is unexplainable. Cite 2017 or 2021, never 2016. | s5 |
| 9 | He et al. ResNet | arXiv:1512.03385, CVPR 2016 | "Deeper neural networks are more difficult to train" - the DEGRADATION problem (training failure, not overfitting); residual reformulation; 152 layers "8x deeper than VGG"; ensemble 3.57% ImageNet top-5, 1st place ILSVRC 2015. | 3.57% is an ENSEMBLE number. "ResNet fixed vanishing gradients" is wrong - BatchNorm had largely handled that; the paper's claim is optimization tractability via identity shortcuts. 1000+ layer runs are CIFAR, not ImageNet. | s4 |
| 10 | PyTorch loop + status | https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html + github.com/pytorch/pytorch/releases | The five moves: forward → loss → backward → step → zero_grad. Verbatim rationale: "Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration." Status Sept 2026: stable = PyTorch 2.14.0 (released 2026-09-02); torch.compile mature standard opt-in since 2.0. | zero_grad order flexible - the invariant is once per iteration before the next backward. 2.14.0 was one day old at verification - re-check before delivery. | s3 |
| 11 | Tabular boundary | Grinsztajn et al. NeurIPS 2022 D&B (arXiv:2207.08815) + Shwartz-Ziv & Armon, Information Fusion 81 (2022, arXiv:2106.03253) | Trees SOTA on medium-sized (~10K) typical tabular; NNs' three challenges verbatim: robust to uninformative features / preserve data orientation / easily learn irregular functions (NNs biased toward smooth solutions). Shwartz-Ziv title claim: "Tabular data: Deep learning is not all you need" - XGBoost + ensembles beat tabular deep models outside their home papers. | Scope medium-sized; gap narrows with scale/pretraining. Shwartz-Ziv exact wording NOT re-fetched - abstract-level claim only, no quote marks. | s6 |
| 12 | AlexNet | Krizhevsky, Sutskever & Hinton, NIPS 2012. https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html (quotes from the PDF) | Verbatim: ILSVRC-2012 "winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry"; "five and six days to train on two GTX 580 3GB GPUs"; 60M parameters. | 15.3% = the 7-CNN ENSEMBLE entry (single CNN 18.2% val) - say "their entry". Landing-page abstract is a stale draft (39.7/18.9) - quote the PDF (37.5/17.0 for ILSVRC-2010). "Halved the error" overshoots - "cut top-5 by over 10 points, ~40% relative" is right. | s4, s6 |

## The simulator: dl-live.js (canon verified in-engine 2026-09-03)

A REAL MLP in vanilla JS: genuine forward/backward passes (full-batch GD, BCE + sigmoid output, tanh/relu hidden), He-scaled seeded init. Two mounts:

**#dl-boundary (session 3):** four deterministic 2D datasets (moons / xor / circle / spiral, 180 train + 60 test), live canvas heatmap = the actual model queried on a 90x90 grid, chunked training (200 epochs per press). Levers: hidden layers 0/1/2, width 4/8/16, rate 0.03/0.3/1.5.

Boundary canon (train / test %, tanh):
| Config | Train | Test | Teaching beat |
|---|---|---|---|
| XOR, no hidden layer, 600ep lr.3 | 62.8 | 48.3 | the model IS a straight line; XOR is why hidden layers exist |
| XOR, 1x8, 600ep lr.3 | 97.8 | 95.0 | one hidden layer of 8 solves it |
| spiral, 1x8, 600ep lr.3 | 55.0 | 55.0 | stuck - looks like a capacity problem... |
| spiral, 1x8, 1200ep lr1.5 | 98.9 | 100 | ...but the SAME net solves it with a hotter rate + more epochs: the capacity was there, the optimizer had not arrived. UAT caveat, made visible |
| spiral, 2x16, 600ep lr1.5 | 100 | 100 | depth + width make the optimization easy |
| moons, 1x8, 600ep lr.3 | 92.8 | 88.3 | easy data, everything works |

**#dl-tabular (session 6):** the SAME seeded 300-customer Mango Lane generator as ensemble-live.js, byte-identical, standardized features, 200/100 split. Bench row shows the ensemble course's tuned booster: 100 / 73.0.

Tabular showdown canon (train / holdout %):
| Config | Train | Holdout | Teaching beat |
|---|---|---|---|
| Logistic regression (no hidden), 400ep | 67.5 | 64.0 | the linear baseline |
| MLP 1x8 tanh, 600ep | 80.5 | 65.0 | a point over linear |
| MLP 2x16 tanh, 600ep | 100 | 60.0 | memorizes 200 rows, WORST on holdout |
| MLP 3x32 overparameterized, 1500ep | 100 | 68-70 (engine-dependent) | biggest net is the BEST net (Belkin-flavored) - still under the trees on every engine. Cross-engine float variation (Math.exp/tanh differ per JS engine; 1500 chaotic epochs amplify it) - TAUGHT on the widget as "determinism has layers" |
| ◆ Tuned gradient booster (bench) | 100 | 73.0 | Grinsztajn, measured on our own data |

## Per-session coverage

| # | Session | Teaches | Sources | Coverage |
|---|---|---|---|---|
| s1 | The neuron | Weights, bias, activation; why nonlinearity; what a layer is; UAT stated WITH its existence-only caveat | 2 | ✓ |
| s2 | Backprop by hand | Chain rule on a scalar graph, micrograd-style; the 1986 representations claim + honest lineage (Linnainmaa/Werbos) | 1, 6 | ✓ |
| s3 | The training loop | Loss, gradient descent, learning rate, epochs; the PyTorch five moves; Karpathy recipe rules; BOUNDARY SIMULATOR + halfway scorecard | 5, 6, 10, 2 | ✓ |
| s4 | Seeing and embedding | Convolution as inductive bias (LeNet), AlexNet + GPUs, ResNet degradation problem; embeddings as learned representations; pointer to the GenAI shelf for transformers | 3, 9, 12 | ✓ |
| s5 | Taming the net | Overfit diagnosis, dropout, early stopping, get-more-data rule, AdamW nuance; random-labels result; double-descent teaser card | 4, 5, 6, 7, 8 | ✓ |
| s6 | When deep learning | The modality map (inductive bias matching structure); TABULAR SHOWDOWN on shared data vs the booster; the honest boundary; FINAL scorecard | 11, 12, 3 | ✓ |

## Hard rails / honesty

- The simulator is REAL (forward/backward passes live); the datasets are the teaching artifice (seeded, stated on-widget).
- Carry the misquote watchlist as course content: backprop 1986 = popularized not invented; UAT = existence only; dropout approximates an ensemble; AdamW is the default, not Adam; double descent does not kill regularization; ResNet = degradation not vanishing gradients; AlexNet 15.3% = the ensemble entry; trees-win scoped to medium tabular.
- s6 must NOT gloat: the showdown's honest reading is Grinsztajn + McElfresh, not "nets are bad" - the same 3x32 net that loses here wins at scale on images/text, and the modality map says why.
- The in-browser MLP is a teaching implementation (full-batch GD, no Adam, no minibatches) - s3 says so and maps each missing piece to the PyTorch loop.

## Not covered by design

- Transformers, attention, LLMs (GenAI shelf); RNNs beyond a mention; GPU programming; PyTorch beyond the canonical loop; the full generalization-theory debate (parked learn-ml-epistemology).

**Re-verify before delivery:** PyTorch stable version (2.14.0 was one day old); Shwartz-Ziv exact wording if ever quote-marked.
