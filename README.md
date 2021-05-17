# CS 188
This repo contains all the projects done 

### Setup
For code to run properly in this class, create a conda environment
```sh
conda create --name cs188 python=3.6
```

After that you can run:
```sh
conda activate cs188
```

To check correctness, you can run this command in each project:
```sh
python3 autograder.py
```

Or you can run a specific question with:
```sh
python autograder.py -q q1
```

Commands use for GUI or test cases will be stored in `commands.txt`. Otherwise, you can click on the project title for a more detailed description of the spec. Or run them all with:
```sh
bash commands.txt
```

## [Project 0: Tutorial](https://inst.eecs.berkeley.edu/~cs188/fa20/project0/)
A basic Python tutorial that refreshes memory on basics of Python and common data structures that will be used heavily in this class.

## [Project 1: Search](https://inst.eecs.berkeley.edu/~cs188/fa20/project1/)
We implemented BFS, DFS, UCS, and A* search to solve Pacman path finding. 

## [Project 2: Multiagent](https://inst.eecs.berkeley.edu/~cs188/fa20/project2/)
Used gametrees to solve Pacman with adversarial agents/ghosts. Implemented minimax, alpha-beta pruning, and expectimax.

## [Project 3: Reinforcement](https://inst.eecs.berkeley.edu/~cs188/fa20/project3/)
Implemented MDPs and RL algorithms. Specifically, Q-learning and value iteration.

## [Project 4: Tracking](https://inst.eecs.berkeley.edu/~cs188/fa20/project4/)
Used Bayes net and partical filtering to solve a game of ghostbusters. Pacman tries to capture ghosts with inference.

## [Project 5: Machine Learning](https://inst.eecs.berkeley.edu/~cs188/fa20/project5/)
Implemented a perceptron algorithm. Then expanded the implementation by building neural networks from scratch to approximate a sine wave, classify MNIST digits, and identify languages.