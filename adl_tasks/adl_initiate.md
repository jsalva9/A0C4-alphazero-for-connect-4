**Applied Deep Learning project: Initiate** 

**Joan Salvà Soler - 12223411**

# A0C4: An implementation of AlphaZero for Connect 4

### Motivation
Throughout my short yet academic and professional experience, one of the topics that
has always fascinated me is the field of Mathematical Optimization. In particular, Combinatorial Optimization
problems lie in the intersection of Mathematical modelling and classical algorithms, and could be considered the 
core of the NP-hard class. 

On the other hand, I was amazed by the recent advances in the field of Deep Reinforcement Learning,
and specially how it has been applied to build the best artificial engines for games such as Go, Chess or Shogi.
In particular, the AlphaZero algorithm is a general Deep Reinforcement algorithm that learns to play these games through
self-play, and has been able to beat the best human players and the best engines with no human knowledge and very 
limited training time.

My fascination for AlphaZero does not only come from its performance, but also because it is a general algorithm that 
I think could be applied to many other contexts, for instance to Combinatorial Optimization problems. 
For example, AlphaZero and its Monte Carlo Tree Search could learn branching strategies for 
Branch-and-Bound algorithms, or it could learn to build heuristics for finding 
feasible solutions to NP-hard problems. That is a topic that I might consider for my Master Thesis.

However, the architectural design of AlphaZero is not trivial. Multiple DNNs are used in combination with a Monte Carlo 
Tree Search algorithm that performs the tree search. It is known to be computationally very expensive, and the 
performance is very sensitive to implementation details. Therefore, I decided to use the Applied Deep Learning
project to get a better understanding of this revolutionary algorithm, and to reproduce it for a simple game: Connect 4 
in a 6x7 board. The fact that this game has already been mathematically solved makes it a good candidate 
for this project, since we can compare the performance of our AlphaZero implementation with the optimal solution. 

### Project overview
The goal of the project is to implement the AlphaZero algorithm for the game of Connect 4 in a 6x7 board. DeepMind 
presented the AlphaZero algorithm in [Silver et. al. (2018)](https://arxiv.org/abs/1712.01815), and is based on
the previous work of AlphaGo Zero and AlphaGo. This previous paper will be the main reference for the project.
DeepMind never released the code for AlphaZero, but many implementations from the community have been published. 
Some references are [Tian et. al. (2022)](https://arxiv.org/abs/1902.04522) or [McGrath et. al. (2021)](https://arxiv.org/abs/2111.09259).
However, our work will not use any existing implementation, but will just use the pseudocode provided in the paper, along
with the insights from the implementation attempts. For the DNNs, we will use existing frameworks such as PyTorch or TensorFlow. The Monte-Carlo 
tree search will be implemented from scratch. The game of [Connect 4](https://wikipedia.org/wiki/Connect_Four) will be 
implemented from scratch as well, which will give us control over the state representation and the rules of the game 
(we might have time to try Connect x in a y*z board). The evaluation of the performance (for the basic C4 rules) will be 
done by comparing the learned policy with the optimal policy. Finally, a UI will allow a user to play against the 
engine. Rather than being a gaming interface (there are many ways to play C4 online), the goal is to provide insights
on the AlphaZero's choices through the visualization of computed state-values and prior probabilities.

The challenges of the project are the following:
 - AlphaZero uses a complex architecture. Understanding its design and figuring out the implementation details will be 
a challenge. The pseudocode provided in the paper is not very detailed, so significant research will be required.
 - The algorithm will be challenging to implement, and good coding practices will be required to make it work.
 - Though Connect 4 is significantly easier than chess, training could still be computationally expensive, 
and we will have to find a way to make it work in a reasonable time.
 - Many hyperparameters will have to be tuned. For example, each DNN could have a different architecture. The
Monte-Carlo tree search will also have hyperparameters to be tuned. Some insights from 
[Schrittweiser et. al. (2018)](https://arxiv.org/abs/1812.06855) on the tuning of the original AlphaGo could be useful.
 - The evaluation of the performance will be challenging, since we will have to find a way to compare the learned policy
with the optimal policy. Some [C++ open library](https://github.com/PascalPons/connect4) implements the optimal policy, 
but that would require us to implement a Python interface to use it.

The learning outcomes of the project are the following:
 - A better understanding of one of the state-of-the-art algorithms in Deep Reinforcement Learning.
 - A better understanding of the Monte-Carlo tree search algorithm, a relevant tree-search algorithm for Combinatorial
Optimization problems.
 - Some hands-on-experience with Deep Learning frameworks such as PyTorch or TensorFlow.
 - Some hands-on-experience with the implementation of a complex learning algorithm.

### Work breakdown
The work will be divided in the following tasks:
 - **Task 1** (5 hours): Implement the game of Connect 4 in a 6x7 board. This will be done from scratch, 
and will include the state representation, the rules of the game, and a basic interface and visualization for 
debugging purposes.
 - **Task 2** (5 hours): Research on the AlphaZero algorithm. The output of this task is a detailed pseudocode
of the algorithm and its functioning: how to train it, how to use it to play...
 - **Task 3** (15 hours): Implementation of the AlphaZero algorithm. 
 - **Task 4** (5 hours): Implementation of the evaluation module. Some research has to be done here on the best
evaluation methodology. Testing the agent against the optimal policy is naturally the best option, but I only know of a 
C++ implementation and would be complicated to connect it with our Python agent. Some other AI libraries for Connect 4 
will be explored, even considering the downside of not being the optimal policy. 
 - **Task 5** (10 hours): Training, hyperparameter tuning, and evaluation. We train the neural networks through self-play, 
and evaluate the resulting agent. Hyperparameter tuning is done.
 - **Task 6** (15 hours): Streamlit or PyGame application that allows to visualize the state-values and the prior 
probabilities for a selection of moves.
 - **Task 7** (5 hours): Preparation of the presentation (slides, demo, video recording...). The code will be documented,
a ReadMe will be created, and the GitHub repository will be made public 

**Total**: 60 hours. 
