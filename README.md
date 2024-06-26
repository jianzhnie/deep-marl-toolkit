<h2 align="center"> MARLToolkit: The  Multi Agent Rainforcement Learning Toolkit</h2>

**MARLToolkit** is a *Multi-Agent Reinforcement Learning Toolkit* based on **Pytorch**.
It provides MARL research community a unified platform for developing and evaluating the new ideas in various multi-agent environments.
There are four core features of **MARLToolkit**.

- it collects most of the existing MARL algorithms widely acknowledged by the community and unifies them under one framework.
- it gives a solution that enables different multi-agent environments using the same interface to interact with the agents.
- it guarantees excellent efficiency in both the training and sampling process.
- it provides trained results, including learning curves and pretrained models specific to each task and algorithm's combination, with finetuned hyper-parameters to guarantee credibility.

## Overview

We collected most of the existing multi-agent environment and multi-agent reinforcement learning algorithms and unified them under one framework based on \[**Pytorch**\] to boost the MARL research.

The MARL baselines include **independence learning (IQL, A2C, DDPG, TRPO, PPO)**, **centralized critic learning (COMA, MADDPG, MAPPO, HATRPO)**, and **value decomposition (QMIX, VDN, FACMAC, VDA2C)** are all implemented.

Popular environments like **SMAC, MaMujoco, and Google Research Football** are provided with a unified interface.

The algorithm code and environment code are fully separated. Changing the environment needs no modification on the algorithm side and vice versa.

|                              Benchmark                               |                                                                    Github Stars                                                                     | Learning Mode | Available Env | Algorithm Type | Algorithm Number | Continues Control  | Asynchronous Interact | Distributed Training |                                       Framework                                        |                                                     Last Update                                                     |
| :------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------: | :-----------: | :------------: | :--------------: | :----------------: | :-------------------: | :------------------: | :------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
|             [PyMARL](https://github.com/oxwhirl/pymarl)              |                 [![GitHub stars](https://img.shields.io/github/stars/oxwhirl/pymarl)](https://github.com/oxwhirl/pymarl/stargazers)                 |      CP       |       1       |       VD       |        5         |                    |                       |                      |                                           \*                                           |         ![GitHub last commit](https://img.shields.io/github/last-commit/oxwhirl/pymarl?label=last%20update)         |
|            [PyMARL2](https://github.com/hijkzzz/pymarl2)             |                [![GitHub stars](https://img.shields.io/github/stars/hijkzzz/pymarl2)](https://github.com/hijkzzz/pymarl2/stargazers)                |      CP       |       1       |       VD       |        12        |                    |                       |                      |                      [PyMARL](https://github.com/oxwhirl/pymarl)                       |        ![GitHub last commit](https://img.shields.io/github/last-commit/hijkzzz/pymarl2?label=last%20update)         |
|      [off-policy](https://github.com/marlbenchmark/off-policy)       |                                    ![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/off-policy)                                    |      CP       |       4       |    IL+VD+CC    |        4         |                    |                       |                      |               [off-policy](https://github.com/marlbenchmark/off-policy)                |    ![GitHub last commit](https://img.shields.io/github/last-commit/marlbenchmark/off-policy?label=last%20update)    |
|       [on-policy](https://github.com/marlbenchmark/on-policy)        |                                    ![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/on-policy)                                     |      CP       |       4       |    IL+VD+CC    |        1         |                    |                       |                      |                [on-policy](https://github.com/marlbenchmark/on-policy)                 |    ![GitHub last commit](https://img.shields.io/github/last-commit/marlbenchmark/on-policy?label=last%20update)     |
| [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms) | [![GitHub stars](https://img.shields.io/github/stars/starry-sky6688/MARL-Algorithms)](https://github.com/starry-sky6688/MARL-Algorithms/stargazers) |      CP       |       1       |    VD+Comm     |        9         |                    |                       |                      |                                           \*                                           | ![GitHub last commit](https://img.shields.io/github/last-commit/starry-sky6688/MARL-Algorithms?label=last%20update) |
|           [EPyMARL](https://github.com/uoe-agents/epymarl)           |         [![GitHub stars](https://img.shields.io/github/stars/uoe-agents/epymarl)](https://github.com/hijkzzz/uoe-agents/epymarl/stargazers)         |      CP       |       4       |    IL+VD+CC    |        10        |                    |                       |                      |                      [PyMARL](https://github.com/oxwhirl/pymarl)                       |       ![GitHub last commit](https://img.shields.io/github/last-commit/uoe-agents/epymarl?label=last%20update)       |
|     [Marlbenchmark](https://github.com/marlbenchmark/on-policy)      |        [![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/on-policy)](https://github.com/marlbenchmark/on-policy/stargazers)        |     CP+CL     |       4       |     VD+CC      |        5         | :heavy_check_mark: |                       |                      | [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) |    ![GitHub last commit](https://img.shields.io/github/last-commit/marlbenchmark/on-policy?label=last%20update)     |
|             [MAlib](https://github.com/sjtu-marl/malib)              |            [![GitHub stars](https://img.shields.io/github/stars/sjtu-marl/malib)](https://github.com/hijkzzz/sjtu-marl/malib/stargazers)            |      SP       |       8       |       SP       |        9         | :heavy_check_mark: |                       |                      |                                           \*                                           |        ![GitHub last commit](https://img.shields.io/github/last-commit/sjtu-marl/malib?label=last%20update)         |
|        [MARLlib](https://github.com/Replicable-MARL/MARLlib)         |        [![GitHub stars](https://img.shields.io/github/stars/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib/stargazers)        |  CP+CL+CM+MI  |      10       |    IL+VD+CC    |        18        | :heavy_check_mark: |  :heavy_check_mark:   |  :heavy_check_mark:  |                  [Ray/RLlib](https://docs.ray.io/en/releases-1.8.0/)                   |    ![GitHub last commit](https://img.shields.io/github/last-commit/Replicable-MARL/MARLlib?label=last%20update)     |
|                                                                      |                                                                                                                                                     |               |               |                |                  |                    |                       |                      |                                                                                        |                                                                                                                     |

CP, CL, CM, and MI represent cooperative, collaborative, competitive, and mixed task learning modes.
IL, VD, and CC represent independent learning, value decomposition, and centralized critic categorization. SP represents self-play.
Comm represents communication-based learning.
Asterisk denotes that the benchmark uses its framework.

## Environment

#### Supported Multi-agent Environments / Tasks

Most of the popular environment in MARL research has been incorporated in this benchmark:

| Env Name                                                          | Learning Mode | Observability | Action Space | Observations |
| ----------------------------------------------------------------- | ------------- | ------------- | ------------ | ------------ |
| [LBF](https://github.com/semitable/lb-foraging)                   | Mixed         | Both          | Discrete     | Discrete     |
| [RWARE](https://github.com/semitable/robotic-warehouse)           | Collaborative | Partial       | Discrete     | Discrete     |
| [MPE](https://github.com/openai/multiagent-particle-envs)         | Mixed         | Both          | Both         | Continuous   |
| [SMAC](https://github.com/oxwhirl/smac)                           | Cooperative   | Partial       | Discrete     | Continuous   |
| [MetaDrive](https://github.com/decisionforce/metadrive)           | Collaborative | Partial       | Continuous   | Continuous   |
| [MAgent](https://www.pettingzoo.ml/magent)                        | Mixed         | Partial       | Discrete     | Discrete     |
| [Pommerman](https://github.com/MultiAgentLearning/playground)     | Mixed         | Both          | Discrete     | Discrete     |
| [MaMujoco](https://github.com/schroederdewitt/multiagent_mujoco)  | Cooperative   | Partial       | Continuous   | Continuous   |
| [GRF](https://github.com/google-research/football)                | Collaborative | Full          | Discrete     | Continuous   |
| [Hanabi](https://github.com/deepmind/hanabi-learning-environment) | Cooperative   | Partial       | Discrete     | Discrete     |

Each environment has a readme file, standing as the instruction for this task, talking about env settings, installation, and some important notes.

## Algorithm

We provide three types of MARL algorithms as our baselines including:

**Independent Learning:**
IQL
DDPG
PG
A2C
TRPO
PPO

**Centralized Critic:**
COMA
MADDPG
MAAC
MAPPO
MATRPO
HATRPO
HAPPO

**Value Decomposition:**
VDN
QMIX
FACMAC
VDAC
VDPPO

Here is a chart describing the characteristics of each algorithm:

| Algorithm                                                                                                                  | Support Task Mode | Need Global State | Action     | Learning Mode        | Type       |
| -------------------------------------------------------------------------------------------------------------------------- | ----------------- | ----------------- | ---------- | -------------------- | ---------- |
| IQL                                                                                                                        | Mixed             | No                | Discrete   | Independent Learning | Off Policy |
| [PG](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) | Mixed             | No                | Both       | Independent Learning | On Policy  |
| [A2C](https://arxiv.org/abs/1602.01783)                                                                                    | Mixed             | No                | Both       | Independent Learning | On Policy  |
| [DDPG](https://arxiv.org/abs/1509.02971)                                                                                   | Mixed             | No                | Continuous | Independent Learning | Off Policy |
| [TRPO](http://proceedings.mlr.press/v37/schulman15.pdf)                                                                    | Mixed             | No                | Both       | Independent Learning | On Policy  |
| [PPO](https://arxiv.org/abs/1707.06347)                                                                                    | Mixed             | No                | Both       | Independent Learning | On Policy  |
| [COMA](https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653)                                                   | Mixed             | Yes               | Both       | Centralized Critic   | On Policy  |
| [MADDPG](https://arxiv.org/abs/1706.02275)                                                                                 | Mixed             | Yes               | Continuous | Centralized Critic   | Off Policy |
| MAA2C                                                                                                                      | Mixed             | Yes               | Both       | Centralized Critic   | On Policy  |
| MATRPO                                                                                                                     | Mixed             | Yes               | Both       | Centralized Critic   | On Policy  |
| [MAPPO](https://arxiv.org/abs/2103.01955)                                                                                  | Mixed             | Yes               | Both       | Centralized Critic   | On Policy  |
| [HATRPO](https://arxiv.org/abs/2109.11251)                                                                                 | Cooperative       | Yes               | Both       | Centralized Critic   | On Policy  |
| [HAPPO](https://arxiv.org/abs/2109.11251)                                                                                  | Cooperative       | Yes               | Both       | Centralized Critic   | On Policy  |
| [VDN](https://arxiv.org/abs/1706.05296)                                                                                    | Cooperative       | No                | Discrete   | Value Decomposition  | Off Policy |
| [QMIX](https://arxiv.org/abs/1803.11485)                                                                                   | Cooperative       | Yes               | Discrete   | Value Decomposition  | Off Policy |
| [FACMAC](https://arxiv.org/abs/2003.06709)                                                                                 | Cooperative       | Yes               | Continuous | Value Decomposition  | Off Policy |
| [VDAC](https://arxiv.org/abs/2007.12306)                                                                                   | Cooperative       | Yes               | Both       | Value Decomposition  | On Policy  |
| VDPPO\*                                                                                                                    | Cooperative       | Yes               | Both       | Value Decomposition  | On Policy  |

*IQL* is the multi-agent version of Q learning.
*MAA2C* and *MATRPO* are the centralized version of A2C and TRPO.
*VDPPO* is the value decomposition version of PPO.
