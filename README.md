Deep reinforcement learning program to generate arbitary quantum states. It is able to learn all driving protocols to generate arbitrary states on a 2D Bloch sphere, embedded in a higher dimensional Hilbertspace. The amazing feature is that it produces all the driving protocols (over a continous Bloch sphere) for all the states in a single run. The learning is performed on all target states at the same time. Based on spinning Up AI deep learning with PPO, implemented in Tensorflow.


Prerequisites:
- OpenAi spinningup https://spinningup.openai.com/ (follow instructions there to install)
- matplotlib for plotting
- Qutip (http://qutip.org/) to run the quantum simulation part
- scipy

To run, simply run the main file. To reproduce the data from the paper, choose from 3 pre-defined templates using the variable predefinedTemplates (line 414 in RunSpinUpNV_reduced.py).
