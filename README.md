# losses
This repo contains pytorch implementations of some oddball losses I have found useful

ApproxNDCGloss, Lambdaloss, NeuralNDCGLoss and RankNet are taken from https://github.com/allegro/allRank/tree/master/allrank/models/losses but are now standalone implementations which can just be imported and used.

UPU and NNPU (in unbiased.py) are inspired by https://arxiv.org/pdf/2103.04683.pdf and are my own implementations.



If you spot mistakes in the implementations and/or better ways to implement them, please don't hesitate to open an issue and report it!

## Usage

For usage, please see the testfile ```test_losses.py```. Please note that the y_true vector should be binary but the y_pred vector does not have to. This means that you can directly feed your NN ouput (logits) into the loss, no need to call softmax or logistic function beforehand!
