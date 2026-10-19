\# Learning rate experiment (MLP, MNIST, SGD, batch size 64, 2 epochs)



| Learning rate | Test accuracy |

|---|---|

| 0.001 | 63.67% |

| 0.01  | 89.19% |

| 0.1   | 96.31% |



Observation: Raising the learning rate from 0.001 to 0.1 increased test accuracy after

2 epochs from 63.67% to 96.31%. At 0.001 each weight update is too small to make much

progress in 2 epochs. A learning rate that is too large should eventually overshoot and

become unstable, so the next test is finding where that happens (e.g. 0.5 or 1.0).

