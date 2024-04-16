class QMixConfig:
    """Configuration class for QMix model.

    QMixConfig contains parameters used to instantiate a QMix model.
    These parameters define the model architecture, behavior, and training settings.

    Args:
        mixing_embed_dim (int, optional): Embedding dimension of the mixing network.
        rnn_hidden_dim (int, optional): Dimension of GRU's hidden state.
        gamma (float, optional): Discount factor in reinforcement learning.
        egreedy_exploration (float, optional): Initial 'epsilon' in epsilon-greedy exploration.
        min_exploration (float, optional): Minimum 'epsilon' in epsilon-greedy.
        target_update_interval (int, optional): Sync parameters to target model after 'target_update_interval' times.
        learning_rate (float, optional): Learning rate of the optimizer.
        min_learning_rate (float, optional): Minimum learning rate of the optimizer.
        clip_grad_norm (float, optional): Clipped value of the global norm of gradients.
        hypernet_layers (int, optional): Number of layers in hypernetwork.
        hypernet_embed_dim (int, optional): Embedding dimension for hypernetwork.
        learner_update_freq (int, optional): Update learner frequency.
        double_q (bool, optional): Use Double-DQN.
        algo_name (str, optional): Name of the algorithm.
    """

    model_type: str = 'qmix'

    def __init__(
        self,
        fc_hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        gamma: float = 0.99,
        egreedy_exploration: float = 1.0,
        min_exploration: float = 0.01,
        target_update_tau: float = 0.05,
        target_update_interval: int = 100,
        learning_rate: float = 0.1,
        min_learning_rate: float = 0.00001,
        clip_grad_norm: float = 10,
        hypernet_layers: int = 2,
        hypernet_embed_dim: int = 64,
        mixing_embed_dim: int = 32,
        learner_update_freq: int = 3,
        double_q: bool = True,
        algo_name: str = 'qmix',
    ) -> None:
        # Network architecture parameters
        self.fc_hidden_dim = fc_hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.hypernet_layers = hypernet_layers
        self.hypernet_embed_dim = hypernet_embed_dim
        self.mixing_embed_dim = mixing_embed_dim

        # Training parameters
        self.gamma = gamma
        self.egreedy_exploration = egreedy_exploration
        self.min_exploration = min_exploration
        self.target_update_tau = target_update_tau
        self.target_update_interval = target_update_interval
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_grad_norm = clip_grad_norm
        self.learner_update_freq = learner_update_freq
        self.double_q = double_q

        # Logging parameters
        self.algo_name = algo_name
