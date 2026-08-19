"""Learning rate schedule: linear warmup followed by cosine decay."""

import math


def get_lr(step: int, config) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    if step >= config.max_steps:
        return config.learning_rate * 0.1

    decay_ratio = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    min_lr = config.learning_rate * 0.1
    return min_lr + coeff * (config.learning_rate - min_lr)
