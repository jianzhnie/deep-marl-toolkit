class QMixConfig:
    """Configuration class for QMix model.

    QMixConfig contains parameters used to instantiate a QMix model.
    These parameters define the model architecture, behavior, and training settings.

    Args:
        scenario (str, optional): The scenario in SMAC environments.
        replay_buffer_size (int, optional): Max number of episodes stored in the replay buffer.
        mixing_embed_dim (int, optional): Embedding dimension of the mixing network.
        rnn_hidden_dim (int, optional): Dimension of GRU's hidden state.
        memory_warmup_size (int, optional): Learning won't start until replay buffer size >= 'memory_warmup_size'.
        gamma (float, optional): Discount factor in reinforcement learning.
        exploration_start (float, optional): Initial 'epsilon' in epsilon-greedy exploration.
        min_exploration (float, optional): Minimum 'epsilon' in epsilon-greedy.
        update_target_interval (int, optional): Sync parameters to target model after 'update_target_interval' times.
        batch_size (int, optional): Training batch size.
        total_steps (int, optional): Total steps for training.
        train_log_interval (int, optional): Log interval for training.
        test_log_interval (int, optional): Log interval for testing.
        test_steps (int, optional): Evaluate the model every 'test_steps' steps.
        learning_rate (float, optional): Learning rate of the optimizer.
        min_learning_rate (float, optional): Minimum learning rate of the optimizer.
        clip_grad_norm (float, optional): Clipped value of the global norm of gradients.
        hypernet_layers (int, optional): Number of layers in hypernetwork.
        hypernet_embed_dim (int, optional): Embedding dimension for hypernetwork.
        update_learner_freq (int, optional): Update learner frequency.
        double_q (bool, optional): Use Double-DQN.
        difficulty (str, optional): Difficulty of the environment.
        algo_name (str, optional): Name of the algorithm.
        log_dir (str, optional): Directory to save logs.
        logger (str, optional): Logger to record logs.
    """

    model_type: str = 'qmix'

    def __init__(
        self,
        scenario: str = '3s_vs_3z',
        replay_buffer_size: int = 5000,
        mixing_embed_dim: int = 32,
        rnn_hidden_dim: int = 64,
        memory_warmup_size: int = 32,
        gamma: float = 0.99,
        exploration_start: float = 1.0,
        min_exploration: float = 0.1,
        update_target_interval: int = 1000,
        batch_size: int = 32,
        total_steps: int = 1000000,
        train_log_interval: int = 5,
        test_log_interval: int = 20,
        test_steps: int = 100,
        learning_rate: float = 0.0005,
        min_learning_rate: float = 0.0001,
        clip_grad_norm: float = 10,
        hypernet_layers: int = 2,
        hypernet_embed_dim: int = 64,
        update_learner_freq: int = 2,
        double_q: bool = True,
        difficulty: str = '7',
        algo_name: str = 'qmix',
        log_dir: str = 'work_dirs/',
        logger: str = 'wandb',
    ) -> None:
        # Environment parameters
        self.scenario = scenario
        self.difficulty = difficulty

        # Network architecture parameters
        self.mixing_embed_dim = mixing_embed_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.hypernet_layers = hypernet_layers
        self.hypernet_embed_dim = hypernet_embed_dim

        # Training parameters
        self.replay_buffer_size = replay_buffer_size
        self.memory_warmup_size = memory_warmup_size
        self.gamma = gamma
        self.exploration_start = exploration_start
        self.min_exploration = min_exploration
        self.update_target_interval = update_target_interval
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.train_log_interval = train_log_interval
        self.test_log_interval = test_log_interval
        self.test_steps = test_steps
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_grad_norm = clip_grad_norm
        self.update_learner_freq = update_learner_freq
        self.double_q = double_q

        # Logging parameters
        self.algo_name = algo_name
        self.log_dir = log_dir
        self.logger = logger
