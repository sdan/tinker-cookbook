# memory-rl: Empirical Information Absorption Rates

This recipe measures the "bit rate" of Reinforcement Learning through a minimal memory test: the environment hides a fixed integer and the policy must recover it.

Recent theoretical discussions ([Ord](https://www.tobyord.com/writing/inefficiency-of-reinforcement-learning), [Li](https://richardli.xyz/post/information-bandwidth-rl/), [Schulman](https://thinkingmachines.ai/blog/lora/)) suggest that standard policy gradient methods with scalar rewards are bottlenecked by ~1 bit per episode. By sweeping reward channels (binary, binned scalar, dense per-bit) and signal sizes, we can empirically measure the **Information Absorption Rate** and compare it to theoretical bounds.

## The Environment

The environment holds a latent secret integer $S \in [0, N-1]$. The agent must output $S$ to maximize reward.

- **Signal size:** $H(S) = \log_2 N$ bits
- **Goal:** Measure episodes ($E$) required to memorize $S$
- **Metric:** $\text{Empirical Bit Rate} = \frac{\log_2 N}{E}$

We compare three channel configurations:

1. **SFT (Baseline):** The agent is told "The secret is 42".
   - *Capacity:* ~$\log_2 N$ bits/example

2. **Scalar RL (Bottleneck):** Single scalar reward at episode end.
   - *Binary:* Correct/Incorrect ($\le 1$ bit/episode)
   - *Binned distance:* Distance quantized into $B$ bins ($\le \log_2 B$ bits/episode)

3. **Dense RL (Control):** Per-bit rewards across multiple steps.
   - *Capacity:* ~$\log_2 N$ bits/episode

## Theoretical Expectations

- **Binary rewards:** Empirical bit rate should be $\ll 1$ for large $N$ due to exploration difficulty
- **Binned rewards:** Learning speed should scale with channel capacity ($\log_2 B$)
- **Dense rewards:** Should approach SFT efficiency, confirming "RL inefficiency" is about reward sparsity, not policy gradients

## Usage

### 1. Supervised Baseline (SFT)

```bash
uv run python -m tinker_cookbook.recipes.memory_rl.sft_train \
    N=64 \
    learning_rate=1e-4 \
    batch_size=32 \
    n_steps=500 \
    wandb_project=memory-rl-sft
```

### 2. Scalar RL (Single-Step)

```bash
# Binary reward
uv run python -m tinker_cookbook.recipes.memory_rl.rl_train \
    env_type=single_step \
    N=64 \
    reward_type=binary \
    learning_rate=4e-5 \
    wandb_project=memory-rl-scalar

# Binned distance (8 bins = 3 bits max)
uv run python -m tinker_cookbook.recipes.memory_rl.rl_train \
    env_type=single_step \
    N=64 \
    reward_type=binned_distance \
    reward_bins=8 \
    learning_rate=4e-5 \
    wandb_project=memory-rl-scalar
```

### 3. Dense RL (Multi-Step)

```bash
uv run python -m tinker_cookbook.recipes.memory_rl.rl_train \
    env_type=multi_step \
    N=64 \
    learning_rate=4e-5 \
    wandb_project=memory-rl-dense
```

## Some early results [wip]

### RL continuous distance reward from secret number

![Screenshot 2025-12-01 at 11:42:29 PM](https://github.com/user-attachments/assets/d132cbff-8ab8-4a72-9044-785ff88735a0)

### RL binary reward from secret number

![Screenshot 2025-12-01 at 11:45:11 PM](https://github.com/user-attachments/assets/6bf8ea62-21a8-4c60-afb6-41f23a25c5cd)

### RL multi-step bit reward from secret bitstring

![Screenshot 2025-12-01 at 11:45:51 PM](https://github.com/user-attachments/assets/40afd8bf-8f10-488b-bf2d-1b5f8ad494fa)


