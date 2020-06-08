Deep reinforcement learning program to generate arbitary quantum states. This program has been used for "Engineering quantum current states with machine learning" (https://arxiv.org/abs/1911.09578).
It is able to learn all driving protocols to generate arbitrary states on a 2D Bloch sphere, embedded in a higher dimensional Hilbertspace. The amazing feature is that it produces all the driving protocols (over a continous Bloch sphere) for all possible target states in a single run of the program. The learning is performed on all target states at the same time. Based on spinning Up AI deep learning with PPO, implemented in Tensorflow.


Prerequisites:
- Anaconda with Python 3.6 is recommended
- OpenAi spinningup https://spinningup.openai.com/ (follow instructions there to install)
- gym
- matplotlib
- Qutip (http://qutip.org/) to simulate quantum systems
- scipy

Execute the main file RunSpinUpNV_reduced.py. Various parameters can be configures in the main file, at around line 404. You can choose from 3 pre-defined templates using the variable predefinedTemplates (line 414 in RunSpinUpNV_reduced.py) that reproduce the main results from the publication.
