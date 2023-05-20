# Progress tracking
Going to use this to document my findings and progress on this project in more detail than what commit messages can provide.

## 5/18/23
Came up with the idea, and researched the difference between different types of machine learning to determine what might best suit my project.

## 5/19/23
Learned that in reinforcement learning there is the actor neural network, which determines what the AI does on each time step, as well as the critic neural network, which calculates the expected reward based on the actor's decision.
This is then compared to the actual reward gained from said action, and the difference is called the loss. Then using back propogation you use the loss value to train the actor's neural network.
https://www.mathworks.com/videos/deep-reinforcement-learning-for-walking-robots--1551449152203.html

lots more topics:
https://www.mathworks.com/discovery/reinforcement-learning.html

Main question right now is how should I go about the projet. Because this most recent matlab example makes it seem like I could do the physics of the guy and the neural network all in one place, which would simplify things
However, I don't like matlab, so it might be worth the extra effort to code everything in visual study, and connect the various code sources via API's, plus a more complicated project looks better on the resume.