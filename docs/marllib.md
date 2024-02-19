# MARLlib

## Intro

MARLlib 是一个基于Ray和其工具包RLlib的综合多智能体强化学习算法库。它为多智能体强化学习研究社区提供了一个统一的平台，用于构建、训练和评估多智能体强化学习算法。

<img src="https://marllib.readthedocs.io/en/latest/_images/marllib_open.png" alt="../_images/marllib_open.png" style="zoom: 33%;" />

> Overview of the MARLlib architecture.\`\`

MARLlib 提供了几个突出的关键特点：

1. **统一算法Pipeline：** MARLlib通过基于Agenty 级别的分布式数据流将多样的算法Pipeline统一起来，使研究人员能够在不同的任务和环境中开发、测试和评估多智能体强化学习算法。
2. **支持各种任务模式：** MARLlib支持协作、协同、竞争和混合等所有任务模式。
3. **与Gym结构一致的接口：** MARLlib提供了一个新的与Gym 相似的环境接口，方便研究人员更加轻松的处理多智能环境。
4. **灵活参数共享策略：** MARLlib提供了灵活且可定制的参数共享策略。

使用 MARLlib，您可以享受到各种好处，例如：

1. **零MARL知识门槛：** MARLlib提供了18个预置算法，并具有简单的API，使研究人员能够在不具备多智能体强化学习领域知识的情况下开始进行实验。
2. **支持所有任务模式：** MARLlib支持几乎所有多智能体环境，使研究人员能够更轻松地尝试不同的任务模式。
3. **可定制的模型架构：** 研究人员可以从模型库中选择他们喜欢的模型架构，或者构建自己的模型。
4. **可定制的策略共享：** MARLlib提供了策略共享的分组选项，研究人员也可以创建自己的共享策略。
5. **访问上千个发布的实验：** 研究人员可以访问上千个已发布的实验，了解其他研究人员如何使用MARLlib。



## MARLlib 框架

## 环境接口

<img src="https://marllib.readthedocs.io/en/latest/_images/marl_env_right.png" alt="../_images/marl_env_right.png"  />

Agent-Environment Interface in MARLlib



MARLlib 中的环境接口支持以下功能：

1. 智能体无关：每个智能体在训练阶段都有隔离的数据
2. 任务无关：多种环境支持一个统一的接口
3. 异步采样：灵活的智能体-环境交互模式

首先，MARLlib 将 MARL 视为单智能体 RL 过程的组合。

其次，MARLlib 将所有十种环境统一为一个抽象接口，有助于减轻算法设计工作的负担。并且该接口下的环境可以是任何实例，从而实现多任务 或者 任务无关的学习。

第三，与大多数现有的 MARL 框架仅支持智能体和环境之间的同步交互不同，MARLlib 支持异步交互风格。这要归功于RLlib灵活的数据收集机制，不同agent的数据可以通过同步和异步的方式收集和存储。

##  工作流

### 第一阶段：预学习

MARLlib 通过实例化环境和智能体模型来开始强化学习过程。随后，根据环境特征生成模拟批次，并将其输入指定算法的采样/训练Pipeline中。成功完成学习工作流程并且没有遇到错误后，MARLlib 将进入后续阶段。

![../_images/rllib_data_flow_left.png](https://marllib.readthedocs.io/en/latest/_images/rllib_data_flow_left.png)

> 预习阶段

### 第二阶段：采样和训练

预学习阶段完成后，MARLlib 将实际工作分配给 Worker 和 Leaner，并在执行计划下安排这些流程以启动学习过程。

在标准学习迭代期间，每个 Worker 使用智能体模型与其环境实例交互以采样数据，然后将数据传递到ReplayBuffer。ReplayBuffer 根据算法进行初始化，并决定如何存储数据。例如，对于on-policy算法，缓冲区是一个串联操作，而对于off-policy算法，缓冲区是一个FIFO队列。

随后，预定义的策略映射功能将收集到的数据分发给不同的智能体。一旦完全收集了一次训练迭代的所有数据，学习器就开始使用这些数据优化策略，并将新模型广播给每个Worker以进行下一轮采样。

![../_images/rllib_data_flow_right.png](https://marllib.readthedocs.io/en/latest/_images/rllib_data_flow_right.png)

采样和训练阶段



### 算法Pipeline

![../_images/pipeline.png](https://marllib.readthedocs.io/en/latest/_images/pipeline.png)

### 独立学习

在 MARLlib 中，由于 RLlib 提供了许多算法，实现独立学习（左）非常简单。要开始训练，可以从 RLlib 中选择一种算法并将其应用到多智能体环境中，与 RLlib 相比，无需额外的工作。尽管 MARL 中的独立学习不需要任何数据交换，但在大多数任务中其性能通常不如中心化训练策略。

### 中心化 Critic

中心化 Critic 是 MARLlib 支持的 CTDE 框架中的两种中心化训练策略之一。在这种方法下，智能体需要在获得策略输出之后但在临界值计算之前相互共享信息。这些共享信息包括 独立观察、动作和全局状态（如果有）。

交换的数据在采样阶段被收集并存储为过渡数据，其中每个过渡数据都包含自收集的数据和交换的数据。然后利用这些数据来优化中心化评价函数和分散式策略函数。信息共享的实现主要是在同策略算法的后处理函数中完成的。对于像 MADDPG 这样的Off-Policy算法，在数据进入训练迭代批次之前会收集其他数据，例如其他智能体提供的动作值。

### 价值分解

在 MARLlib 中，价值分解（VD）是另一类中心化训练策略，与中心化评价者的不同之处在于需要智能体共享信息。具体来说，仅需要在智能体之间共享预测的 Q 值或临界值，并且根据所使用的算法可能需要额外的数据。例如，QMIX 需要全局状态来计算混合 Q 值。

VD 的数据收集和存储机制与中心化Critic的数据收集和存储机制类似，智能体在采样阶段收集和存储转换数据。联合Q学习方法（VDN、QMIX）基于原始PyMARL，五种VD算法中只有FACMAC、VDA2C和VDPPO遵循标准RLlib训练流程。

## 关键组件

### 数据收集前的处理

MARL 算法采用 中心化 训练和分散执行（CTDE）范式，需要在学习阶段在Agent之间共享信息。在 QMIX、FACMAC 和 VDA2C 等值分解算法中，总 Q 或 V 值的计算需要Agent提供各自的 Q 或 V 值估计。相反，基于中心化评价的算法，如 MADDPG、MAPPO 和 HAPPO，需要Agent共享他们的观察和动作数据，以确定中心化评价值。后处理模块是Agent与同伴交换数据的理想位置。对于中心化评价算法，Agent可以从其他Agent获取附加信息来计算中心化评价值。另一方面，对于值分解算法，智能体必须向其他智能体提供其预测的 Q 或 V 值。此外，后处理模块还负责使用 GAE 或 N 步奖励调整等技术来计算各种学习目标。

![../_images/pp.png](https://marllib.readthedocs.io/en/latest/_images/pp.png)

数据收集前的后处理

### 批量学习前的后处理

在 MARL 算法的背景下，并非所有算法都可以利用后处理模块。其中一个例子是像 MADDPG 和 FACMAC 这样的Off-Policy算法，它们面临着ReplayBuffer中过时数据无法用于当前训练交互的挑战。为了应对这一挑战，我们实现了一个额外的“批量学习之前”功能，以在采样批次进入训练循环之前准确计算当前模型的 Q 或 V 值。这确保了用于训练的数据是最新且准确的，从而提高了训练效果。

![../_images/pp_batch.png](https://marllib.readthedocs.io/en/latest/_images/pp_batch.png)

批量学习前的后处理

### 中心化价值函数

在中心化的评价家Agent模型中，传统的仅基于Agent自我观察的价值函数被能够适应算法要求的中心化 critic所取代。中心化 critic 负责处理从其他Agent收到的信息并生成 中心化 值作为输出。

### 混合价值函数

在价值分解Agent模型中，保留了原来的价值函数，但引入了新的混合价值函数以获得总体混合值。混合功能灵活，可根据用户要求定制。目前支持VDN和QMIX混合功能。

### 异构优化

在异构优化中，各个Agent参数是独立更新的，因此，策略函数不会在不同Agent之间共享。然而，根据算法证明，顺序更新Agent的策略并设置与丢失相关的传票的值可以导致任何正更新的增量求和。

为了保证算法的增量单调性，利用信任域来获得合适的参数更新，就像HATRPO算法中的情况一样。为了在考虑计算效率的同时加速策略和评价者更新过程，HAPPO 算法中采用了近端策略优化技术。

![../_images/hetero.png](https://marllib.readthedocs.io/en/latest/_images/hetero.png)

异构Agent评价优化

### 策略映射

策略映射在标准化多智能体强化学习 (MARL) 环境的接口方面发挥着至关重要的作用。在 MARLlib 中，策略映射被实现为具有层次结构的字典。顶级键代表场景名称，第二级键包含组信息，四个附加键（**description**、**team_prefix**、 **all_agents_one_policy**和**one_agent_one_policy**）用于定义各种策略设置。 team_prefix键根据Agent的名称**对**Agent进行分组，而最后两个键指示完全共享或非共享策略策略是否适用于给定场景。利用策略映射方法来初始化策略并将其分配给不同的智能体，并且每个策略仅使用其相应策略组中的智能体采样的数据来训练。