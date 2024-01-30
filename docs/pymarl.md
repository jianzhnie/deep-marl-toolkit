# Pymarl

## 支持的算法

- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)

## 代码结构

```python
├── components
│   ├── action_selectors.py
│   ├── episode_buffer.py
│   ├── epsilon_schedules.py
│   └── transforms.py
├── controllers
│   ├── basic_controller.py
├── envs
│   └── multiagentenv.py
├── learners
│   ├── coma_learner.py
│   ├── q_learner.py
│   └── qtran_learner.py
├── main.py
├── modules
│   ├── agents
│   │   └── rnn_agent.py
│   ├── critics
│   │   ├── coma.py
│       ├── qmix.py
│       ├── qtran.py
│       └── vdn.py
├── runners
│   ├── episode_runner.py
│   └── parallel_runner.py
├── run.py
└── utils
    ├── dict2namedtuple.py
    ├── logging.py
    ├── rl_utils.py
    └── timehelper.py
```

# Pymarl2

## 支持的算法

Value-based Methods:

- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [**Qatten**: Qatten: A general framework for cooperative multiagent reinforcement learning](https://arxiv.org/abs/2002.03939)
- [**QPLEX**: Qplex: Duplex dueling multi-agent q-learning](https://arxiv.org/abs/2008.01062)
- [**WQMIX**: Weighted QMIX: Expanding Monotonic Value Function Factorisation](https://arxiv.org/abs/2006.10800)

Actor Critic Methods:

- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VMIX**: Value-Decomposition Multi-Agent Actor-Critics](https://arxiv.org/abs/2007.12306)
- [**LICA**: Learning Implicit Credit Assignment for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2007.02529)
- [**DOP**: Off-Policy Multi-Agent Decomposed Policy Gradients](https://arxiv.org/abs/2007.12322)
- [**RIIT**: Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning.](https://arxiv.org/abs/2102.03479)

## 主要特性

- Value function clipping (clip max Q values for QMIX)
- Value Normalization
- Reward scaling
- Reward Clipping
- Observation Normalization
- Large Batch Size
- N-step Returns
- Rollout Process Number
- e-greedy annealing steps
- Death Agent Masking

## 代码结构

```
├── components
│   ├── action_selectors.py
│   ├── episode_buffer.py
│   ├── epsilon_schedules.py
│   ├── segment_tree.py
│   └── transforms.py
├── controllers
│   ├── basic_central_controller.py
│   ├── basic_controller.py
│   ├── conv_controller.py
│   ├── dop_controller.py
│   ├── lica_controller.py
│   ├── n_controller.py
│   └── ppo_controller.py
├── envs
│   ├── gfootball
│   │   ├── FootballEnv.py
│   ├── matrix_game
│   │   └── one_step_matrix_game.py
│   ├── multiagentenv.py
│   ├── stag_hunt
│   │   └── stag_hunt.py
│   └── starcraft
│       ├── smac_maps.py
│       └── StarCraft2Env.py
├── learners
│   ├── coma_learner.py
│   ├── dmaq_qatten_learner.py
│   ├── fmac_learner.py
│   ├── lica_learner.py
│   ├── max_q_learner.py
│   ├── nq_learner.py
│   ├── offpg_learner.py
│   ├── policy_gradient_v2.py
│   ├── ppo_learner.py
│   ├── q_learner.py
│   └── qtran_learner.py
├── main.py
├── modules
│   ├── agents
│   │   ├── atten_rnn_agent.py
│   │   ├── central_rnn_agent.py
│   │   ├── conv_agent.py
│   │   ├── ff_agent.py
│   │   ├── mlp_agent.py
│   │   ├── noisy_agents.py
│   │   ├── n_rnn_agent.py
│   │   ├── rnn_agent.py
│   │   └── rnn_ppo_agent.py
│   ├── critics
│   │   ├── centralv.py
│   │   ├── coma.py
│   │   ├── fmac_critic.py
│   │   ├── lica.py
│   │   └── offpg.py
│   ├── layer
│   │   └── self_atten.py
│   └── mixers
│       ├── dmaq_general.py
│       ├── dmaq_si_weight.py
│       ├── nmix.py
│       ├── qatten.py
│       ├── qmix_central_no_hyper.py
│       ├── qmix.py
│       ├── qtran.py
│       └── vdn.py
├── run
│   ├── dop_run.py
│   ├── on_off_run.py
│   ├── per_run.py
│   └── run.py
├── runners
│   ├── episode_runner.py
│   └── parallel_runner.py
└── utils
    ├── dict2namedtuple.py
    ├── logging.py
    ├── noisy_liner.py
    ├── rl_utils.py
    ├── th_utils.py
    ├── timehelper.py
    └── value_norm.py
```

# Epymarl

EPyMARL 是一个用于训练协作式多智能体深度强化学习 (MARL) 算法库。EPyMARL 扩展了PyMARL 代码库，其中包括多种 MARL 算法，例如 IQL、COMA、VDN、QMIX 和 QTRAN；然而，原始的 PyMARL 仅支持星际争霸环境，并且 PyMARL 中的许多实现细节都是固定的，不能轻易更改。例如，在 PyMARL 中，所有智能体之间共享参数，策略网络的第一层是 RNN，并且对目标网络使用Hard Update。EPyMARL 代码库扩展了 PyMARL,  添加了与OpenAI Gym 的兼容性，包括其他算法（IA2C、IPPO、MADDPG、MAA2C​​、MAPPO）的支持，开发很多新的特性如 无参数共享、硬/软更新、奖励标准化）等，并包括超参数搜索代码。

## 多智能体强化学习算法

考虑一个包含多个智能体的环境，这些智能体需要合作才能实现目标（形式化为Dec-POMDP）。在每个时间步，每个智能体都可以访问对环境的部分观察，并且每个智能体根据其观察历史，基于其行为策略采取行动。环境接收联合动作（所有智能体的动作），并向每个智能体提供新的观察，并在所有智能体之间共享奖励。合作 MARL 的目标是计算每个智能体的策略，以最大化每一局折扣奖励的预期总和。

## 支持的算法

### Independent Learning Algorithms (独立学习算法)

在这一类算法中，每个智能体都是独立训练的，忽略了环境中存在的其他智能体。

- IQL：在 IQL \[10\]中，每个智能体都根据其轨迹使用 DQN 算法进行训练。
- IA2C：在 IA2C 中，每个智能体根据其轨迹使用 A2C 算法进行训练。
- IPPO：在 IPPO 中，每个智能体根据其轨迹使用 PPO 算法进行训练。

### Centralised Policy Gradient Algorithms (集中式策略梯度算法)

这类算法包括 actor-critic 算法，其中 Actor 是分散的（只以每个Actor的轨迹为条件），而 Critic 是集中的，并以所有Actor的联合轨迹为条件计算联合状态值函数（V 值）或联合状态-行动值函数（Q 值）。

- MADDPG：MADDPG是 DDPG 算法的多智能体版本，在该算法中，Critic 被集中训练以近似联合状态-行动值。
- COMA：COMA 是一种Actor-Critic算法，Critic 计算集中的状态-行动值函数。与传统的 Actor 相比，COMA 的主要贡献在于它使用了一种改进的优势估算方法，允许根据共享奖励进行信用分配。
- MAA2C：MAA2C 是 A2C 算法的多智能体版本，其中的 Critic 是以所有智能体的联合轨迹为条件集中训练的状态值函数。
- MAPPO：MAPPO 是 PPO 算法的多智能体版本，其中的 Critic 是一个集中训练的状态值函数，以所有智能体的联合轨迹为条件。

### Value Decomposition Algorithms (价值分解算法)

这类算法试图根据每个智能体的贡献，将智能体获得的共享奖励分解成单个效用。

- VDN：VDN 是一种基于 IQL 的算法。每个智能体都有一个近似其 Q 值的网络。将所有智能体的 Q 值相加，计算出联合行动的 Q 值。联合行动的 Q 值使用标准 Q 学习算法进行训练。
- QMIX：QMIX 是另一种值分解算法，它使用具有可学习参数的混合神经网络来逼近联合行动的 Q 值，而不是将单个 Q 值相加。与 VDN 相比，它允许对共享奖励进行更广泛的因式分解。
- Qatten：Qatten 是一种基于 QMIX 的算法，它使用注意力机制来计算每个智能体的贡献，并将其应用于 QMIX 的混合网络。

## 代码结构

EPyMARL 和 PyMARL 是由多个文件和数千行代码组成的大型代码库。因此，最初很难用于实现新算法。在本节中，将介绍 EPyMARL 的基本代码结构，以及如何将新算法纳入代码库。下面，将介绍 EPyMARL 代码库的主要文件夹结构。

```shell
|-- components (基本的 RL 组件， 如 Replay Buffer等)
|-- config (配置文件)
|   |-- algs (for the algorithms)
|   |-- envs (for the environments)
|-- controllers (动作选择的控制器)
|-- envs (环境封装)
|-- learners (用于训练不同算法的代码)
|-- modules
|   |-- agents (策略网络的网络结构)
|   |-- critics (Critics的网络结构)
|   |-- mixers (Mixer的网络结构)
|-- pretrained
|-- runners (用于智能体和环境交互的代码)
|-- utils
```

该agents文件夹包含为智能体执行 动作 选择的网络体系结构。在 EPyMARL 中，有两种不同的网络实现；一种是在所有智能体之间共享参数，另一种是不共享参数。

该controllers文件夹包含实现完整动作选择pipeline的文件。该代码构建网络的输入向量，更新 RNN 的隐藏状态（如果有），并执行动作选择策略（e-greedy、greedy 和 soft）。此外，它还初始化用于动作选择的智能体网络。

该 learners 文件夹包含实现所有网络训练的代码。首先，它在构造函数中初始化仅在训练期间使用的任何模型（例如Critic或Mixer）。它的主要功能是实现所有网络的训练（包括智能体网络和仅在训练期间使用的网络）。默认情况下，代码库在每集结束时执行梯度更新。因此，学习者类的函数 train 接收一批剧集作为输入，并实现所有损失的计算（例如演员和评论家的损失）并执行梯度步骤。

该 runner文件夹包含智能体与环境之间交互的两种不同实现。第一个是 RL 的经典实现，其中智能体与环境的单个实例交互。第二个实现环境的多个并行实例，并且在每个时间步骤，智能体与所有这些实例进行交互。

该 components 文件夹包含几个实现 RL 基本功能的文件，例如经验回放、不同的动作选择方法（例如 e-greedy 或软策略）、奖励标准化等。

详细的代码结构如下：

```shell
├── components
│   ├── action_selectors.py
│   ├── episode_buffer.py
│   ├── epsilon_schedules.py
│   ├── standarize_stream.py
│   └── transforms.py
├── controllers
│   ├── basic_controller.py
│   ├── maddpg_controller.py
│   └── non_shared_controller.py
├── envs
│   └── multiagentenv.py
├── learners
│   ├── actor_critic_learner.py
│   ├── actor_critic_pac_dcg_learner.py
│   ├── actor_critic_pac_learner.py
│   ├── coma_learner.py
│   ├── maddpg_learner.py
│   ├── ppo_learner.py
│   ├── q_learner.py
│   └── qtran_learner.py
├── main.py
├── modules
│   ├── agents
│   │   ├── rnn_agent.py
│   │   ├── rnn_feature_agent.py
│   │   └── rnn_ns_agent.py
│   ├── critics
│   │   ├── ac_ns.py
│   │   ├── ac.py
│   │   ├── centralV_ns.py
│   │   ├── centralV.py
│   │   ├── coma_ns.py
│   │   ├── coma.py
│   │   ├── maddpg_ns.py
│   │   ├── maddpg.py
│   │   ├── mlp.py
│   │   ├── pac_ac_ns.py
│   │   ├── pac_ac.py
│   │   └── pac_dcg_ns.py
│   └── mixers
│       ├── qmix.py
│       ├── qtran.py
│       └── vdn.py
├── pretrained
│   ├── adversary.py
│   ├── adv_params.pt
│   ├── ddpg.py
│   ├── prey_params.pt
│   └── tag.py
├── runners
│   ├── episode_runner.py
│   └── parallel_runner.py
├── run.py
├── search.config.example.yaml
├── search.py
└── utils
    ├── dict2namedtuple.py
    ├── logging.py
    ├── rl_utils.py
    └── timehelper.py
```
