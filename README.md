<h1 align="center">Playing Pong with Deep Reinforcement Learning:grinning:</h1>
Deep learning model is presented in this project to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.

<pre>
<p align="center">
  <img src="./ImagesGIFs/Pong.gif" width="240" height="180" border="10">
</p>
</pre>


## Introduction

Learning to control agents directly from high-dimensional sensory inputs like vision and speech is one of the long-standing challenges of reinforcement learning (RL). Most successful RL applications that operate on these domains have relied on hand-crafted features combined with linear value functions or policy representations. Clearly, the performance of such systems heavily relies on the quality of the feature representation.Recent advances in deep learning have made it possible to extract high-level features from raw sensory data, leading to breakthroughs in computer vision and speech recognition. These methods utilize a range of neural network architectures, including convolutional networks, multilayer perceptrons, restricted Boltzmann machines, and recurrent neural networks, and have exploited both supervised and unsupervised learning. It seems natural to ask whether similar techniques could also be beneficial for RL with sensory data.

However reinforcement learning presents several challenges from a deep learning perspective. Firstly, most successful deep learning applications to date have required large amounts of handlabelled training data. RL algorithms, on the other hand, must be able to learn from a scalar reward signal that is frequently sparse, noisy and delayed. The delay between actions and resulting rewards, which can be thousands of timesteps long, seems particularly daunting when compared to the direct
association between inputs and targets found in supervised learning. Another issue is that most deep learning algorithms assume the data samples to be independent, while in reinforcement learning one typically encounters sequences of highly correlated states. Furthermore, in RL the data distribution changes as the algorithm learns new behaviours, which can be problematic for deep learning methods that assume a fixed underlying distribution.

This project demonstrates that a convolutional neural network can learn successful control policies from raw video data in the Pong RL environment. The network is trained with a variant of the Q-learning algorithm, with stochastic gradient descent to update the weights.To alleviate the problems of correlated data and non-stationary distributions, we use an experience replay mechanism which randomly samples previous transitions, and thereby smooths the training distribution over many past behaviors. The goal is to create a single neural network agent that is able to successfully learn to play pong. The network was not provided with any game-specific information or hand-designed visual features, and was not privy to the internal state of the emulator; it learned from nothing but the video input, the reward and terminal signals, and the set of possible actions—just as a human player would. In this project, Deep learning model is built to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.

## Algorithm
<pre>
<p align="center">
  <img src="./ImagesGIFs/8.PNG">
</p>
</pre>
> This algorithm is **model-free**: it solves the reinforcement learning task directly using samples from the emulator, without explicitly constructing an estimate of emulator. It is also **off-policy**: it learns about the greedy strategy while following a behaviour distribution that ensures adequate exploration of the state space. In practice, the behaviour distribution is often selected by an `EPSILON-Greedy Strategy` that follows the greedy strategy with probability `1 - EPSILON` and selects a random action with probability `EPSILON`.
 
## Image Preprocessing
Working directly with raw RL Pong frames, which are `640 × 480` pixel images with a `128` color palette, can be computationally demanding, so we apply a basic preprocessing step aimed at reducing the input dimensionality. The raw frames are preprocessed by first converting their `RGB` representation to `gray-scale` and down-sampling it to a `80 × 80` image.For the experiments in this paper, the function `φ` from `algorithm 1` applies this preprocessing to the `last 4 frames` of a history and stacks them to produce the input to the Q-function.

## Model Architecture
There are several possible ways of parameterizing Q using a neural network. Since Q maps history-action pairs to scalar estimates of their Q-value, the history and the action have been used as inputs to the neural network by some previous approaches.

:worried: The main drawback of this type of architecture is that a separate forward pass is required to compute the Q-value of each action, resulting in a cost that scales linearly with the number of actions.

:smiley: We instead use an architecture in which there is a separate output unit for each possible action, and only the state representation is an input to the neural network. The outputs correspond to the predicted Q-values of the individual action for the input state. The main advantage of this type of architecture is the ability to compute Q-values for all possible actions in a given state with only a single forward pass through the network.

<pre>
<p align="center">
  <img src="./ImagesGIFs/7.png">
</p>
</pre>


The exact architecture is shown schematically in above Figure.
- The **input to the neural network** consists of an `80 × 80 × 4` image produced by the preprocessing map `φ` .
- The **first hidden layer** convolves `32 filters` of `8 × 8` with `strides 4` with the input image and applies a rectifier nonlinearity.
- The **second hidden layer** convolves `64 filters` of `4 × 4` with `strides 2` again followed by a rectifier nonlinearity.
- This is followed by a **third convolutional layer** that convolves `64 filters` of `3 × 3` with `strides 1` followed by a rectifier.
- Each convolutional layer is followed by `2 × 2` **max pooling layer**.
- The **final hidden layer** is fully-connected and consists of `256` rectifier units.
- The **output layer** is a fully connected linear layer with a single output for each valid action. The number of valid actions in Pong is `3`.

## List of Hyperparameters and their values

The values of all the hyperparameters were selected by performing an informal search on the games Pong, Breakout, Seaquest, Space Invaders and Beam Rider. We did not perform a systematic grid search owing to the high computational cost, although it is conceivable that even better results could be obtained by systematically tuning the hyperparameter values.

| Hyperparamter           | Value    | Description                                                                                                                                       |
| ----------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Minibatch size          | `100`    | The number of training cases over which each stochastic gradient descent update is computed.                                                      |
| Replay memory size      | `500000` | SGD updates are sampled from this number of most recent frames.                                                                                   |
| Agent history length    | `4`      | The number of most recent frames experienced by the agent that is given as input to the Q- Network.                                               |
| Discount factor         | `0.99`   | Discount factor gamma used in Q-Learning update.                                                                                                  |
| Learning rate           | `1e-6`   | The learning rate used by adam optimizer.                                                                                                         |
| Initial exploration     | `1.00`   | The initial value of `EPSILON` in `EPSILON-greedy exploration`.                                                                                   |
| Final exploration       | `0.1`    | The final value of `EPSILON` in `EPSILON-greedy exploration`.                                                                                     |
| Final exploration frame | `500000` | The number of frames over which the initial value of `EPSILON` is linearly annealed to it's final value.                                          |
| Replay start size       | `50000`  | Uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory. |

## Installation Dependencies
Download `tensorflow 1.9.0` from  [tensorflow-windows-wheel](https://github.com/fo40225/tensorflow-windows-wheel/tree/master/1.9.0/py27/CPU/sse2).
```
python 2.7
tensorflow 1.9.0
pygame 1.9.6
opencv-python 4.2.0
```
## How to Run?
```
git clone https://github.com/Junth19/Playing-Pong-with-Deep-Reinforcement-Learning.git
cd Playing-Pong-with-Deep-Reinforcement-Learning
python DQN Brain.py
```

## Results
Better results were achieved after approximately `1.38 million-time steps`, which corresponds to about `48 hours` of game time. Qualitatively, the network played at the level of an experienced human player, usually beating the game with a score of `20 − 2` .

**Youtube Result:** [DQN Playing Pong](https://www.youtube.com/watch?v=OGb382EyOpg).

<a href="https://www.youtube.com/watch?v=OGb382EyOpg" target="_blank"><img src="./ImagesGIFs/2.PNG"
alt="DQN Playing Pong" width="240" height="180" border="10" /></a>

## ScreenShots
<p align="center">
  <img src="./ImagesGIFs/2.PNG" width="240" height="180" border="10">            <img src="./ImagesGIFs/4.PNG" width="240" height="180" border="10"> 
</p>

<p align="center">
  <img src="./ImagesGIFs/3.PNG" width="240" height="180" border="10">            <img src="./ImagesGIFs/1.PNG" width="240" height="180" border="10">
</p>

# References
1. Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David, Rusu, Andrei A, Veness, Joel,
Bellemare, Marc G, Graves, Alex, Riedmiller, Martin, Fidjeland, Andreas K, Ostrovski,
Georg, et al. [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236).
2. Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David, Graves, Alex, Antonoglou, Ioannis, Wier-stra, Daan, and Riedmiller, Martin. [Playing atari with deep reinforcement learning](https://arxiv.org/abs/1312.5602).
3. Guest Post: [Demystifying Deep Reinforcement Learning](https://www.intel.ai/demystifying-deep-reinforcement-learning/#gs.1afy66)
