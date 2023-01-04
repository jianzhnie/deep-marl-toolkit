Config = {
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
    'min_exploration': 0.1,
    'update_target_interval': 1000,
    'batch_size': 32,
    'total_steps': 1000000,
    'train_log_interval': 5,  # log every 10 episode
    'test_log_interval': 20,  # log every 100 epidode
    'clip_grad_norm': 10,
    'hypernet_layers': 2,
    'hypernet_embed_dim': 64,
    'update_learner_freq': 2,
    'double_q': True,
    'difficulty': '7',
    'algo': 'idqn',
    'log_dir': 'work_dirs/',
    'logger': 'wandb'
}
