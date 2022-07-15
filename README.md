# RL_Project
Folder for the project of the RL course

- Valentina Blasone
- Giulia Marchiori Pietrosanti

______________________________

The project tackles the problem of the "LunarLander-v2", an environment of OpenAI gym. The **goal** is to land a shuttle on the moon.

## Start and termination
**Start**
- The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

**Termination**
1. The lander crashes (the lander body gets in contact with the moon);
2. The lander gets outside of the viewport (x coordinate is greater than 1);
3. The lander is not awake, i.e. it doesn’t move and doesn’t collide with any other body.

## States and actions
**States** - continuous
- 8 variables: 6 continuous ($x$, $y$, $v_x$, $v_y$, $\theta$, $\omega$) and 2 boolean.

**Actions** - discrete
- 4 actions: do nothing, fire right (go left), fire main, fire left (go right)
