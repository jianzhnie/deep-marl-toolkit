class ComaConfig:
    """Configuration class for Coma model.

    ComaConfig contains parameters used to instantiate a Coma model.
    These parameters define the model architecture, behavior, and training settings.

    Args:
        rnn_hidden_dim (int, optional): Dimension of GRU's hidden state.
        gamma (float, optional): Discount factor in reinforcement learning.
        td_lambda (float, optional): Lambda in TD(lambda).
        exploration_start (float, optional): Initial 'epsilon' in epsilon-greedy exploration.
        min_exploration (float, optional): Minimum 'epsilon' in epsilon-greedy.
        update_target_interval (int, optional): Sync parameters to target model after 'update_target_interval' times.
        learning_rate (float, optional): Learning rate of the optimizer.
        min_learning_rate (float, optional): Minimum learning rate of the optimizer.
        clip_grad_norm (float, optional): Clipped value of the global norm of gradients.
        update_learner_freq (int, optional): Update learner frequency.
        double_q (bool, optional): Use Double-DQN.
        algo_name (str, optional): Name of the algorithm.
    """

    model_type: str = 'coma'

    def __init__(
        self,
        fc_hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        gamma: float = 0.99,
        td_lambda: float = 0.8,
        exploration_start: float = 1.0,
        min_exploration: float = 0.1,
        update_target_interval: int = 1000,
        learning_rate: float = 0.0005,
        min_learning_rate: float = 0.0001,
        clip_grad_norm: float = 10,
        update_learner_freq: int = 2,
        double_q: bool = True,
        algo_name: str = 'coma',
    ) -> None:
        # Network architecture parameters
        self.fc_hidden_dim = fc_hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # Training parameters
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.exploration_start = exploration_start
        self.min_exploration = min_exploration
        self.update_target_interval = update_target_interval
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_grad_norm = clip_grad_norm
        self.update_learner_freq = update_learner_freq
        self.double_q = double_q

        # Logging parameters
        self.algo_name = algo_name
