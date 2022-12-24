"""
QMixConfig:
    'scenario': (str) The scenario in smac environments. All scenarios: ['3m',
        '8m', '25m', '5m_vs_6m', '8m_vs_9m', '10m_vs_11m', '27m_vs_39m',
        'MMM', 'MMM2', '2s3z', '3s5z', '3s5z_vs_3s6z', '3s_vs_3z',
        '3s_vs_4z', '3s_vs_5z', '1c3s5z', '2m_vs_1z', 'corridor',
        '6h_vs_8z', '2s_vs_1sc', 'so_many_baneling', 'bane_vs_bane',
        '2c_vs_64zg']
    'replay_buffer_size': (int) Max episode number to be stored in the replay buffer.
    'mixing_embed_dim': (int) Embed dim of the mixing network.
    'rnn_hidden_dim': (int) Dim of GRU's hidden state.
    'memory_warmup_size': (int) The learning process will not start until current
        replay buffer size >= 'memory_warmup_size'.
    'gamma': (float) Discount factor in reinforcement learning.
    'exploration_start': (float) Initial 'epsilon' in epsilon-greedy based exploration.
    'min_exploration': (float) Min 'epsilon' in epsilon-greedy.
    'update_target_interval': (int) Synchronize parameters to the target model after
        the model has been learned 'update_target_interval' times.
    'batch_size': (int) Training batch_size.
    'total_steps': (int) Total steps for training.
    'test_steps': (int) Evaluate the model every 'test_steps' steps.
    'clip_grad_norm': (float) clipped value of global norm of gradients.
    'hypernet_layers': (int; 1 or 2) Number of layers in hypernetwork.
    'hypernet_embed_dim' (int, only make sense when 'hypernet_layers'==2)
    'double_q' (bool, True or False) Double-DQN.
    'difficulty': (str) Difficulty of the environment. Max Value: "7" (very difficult)
"""

QMixConfig = {
    'project': 'StarCraft-II',
    'scenario': '3s_vs_3z',
    'replay_buffer_size': 5000,
    'mixing_embed_dim': 32,
    'rnn_hidden_dim': 64,
    'learning_rate': 0.0005,
    'min_learning_rate': 0.0001,
    'memory_warmup_size': 32,
    'gamma': 0.99,
    'exploration_start': 1.0,
    'min_exploration': 0.05,
    'update_target_interval': 20,
    'batch_size': 32,
    'total_episode': 100000,
    'train_log_interval': 10,  # log every 10 episode
    'test_log_interval': 50,  # log every 100 epidode
    'clip_grad_norm': 10,
    'hypernet_layers': 2,
    'hypernet_embed_dim': 64,
    'update_learner_freq': 2,
    'double_q': True,
    'difficulty': '7',
    'algo': 'qmix',
    'log_dir': 'work_dirs/',
    'logger': 'wandb'
}
