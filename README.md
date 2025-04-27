# ATC-lite

Gymnasium enviornment and Training code for the "Air Traffic Control (ATC) problem".

## Overview

The problem is defined as follows:

> The agent must learn to navigate the aircraft in a given airspace, avoiding collisions and ensuring safe distances between aircraft. More than an ATC planning problem, it is a navigation problem, where the agent must learn to navigate the aircraft in a given airspace, avoiding collisions and ensuring safe distances between aircraft.

The problem has been trained for 2 different scenarios:

- A real world scenario, where the agent must learn to navigate a singular aircraft in the airspace of an actual airport. We have selected New Orleans Lakefront Airport (KNEW) as our real world scenario. The agent must learn to navigate the aircraft in the airspace of the airport, avoiding collisions and ensuring safe distances between aircraft. Using ADS-B data, we have created a simulation of an actual aircraft's trajectory, which is used to compare the agent's performance against the actual aircraft's trajectory.
- A simulated scenario, where the agent must learn how to work with 2 aircraft at the same time. The agent must learn to land them after one another.

We have used the Proximal Policy Optimization (PPO) algorithm to train the agent, along with Curriculum Learning. The agent is trained using a reward function that is based on the distance between the aircraft and the runway, as well as the distance between the aircraft and other aircraft in the airspace. The agent is trained to maximize the reward function, while minimizing the distance to the runway and avoiding collisions with other aircraft.

## File Structure

```bash
.
├── README.md
├── atc
```
