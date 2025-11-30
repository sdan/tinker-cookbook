> # Empricial study on the ability of RL to memorize information
> 
> RL memory test. Our post on LoRA presented theoretical arguments on the rate of information acquisition by both SFT and RL. You can set up a toy environment where RL must learn a completely random number sequence, to compare the empirical learning rate under various reward functions to the theoretical estimate.
> 
> [LoRA without regret](https://thinkingmachines.ai/blog/lora/) post makes some theoretical arguments about the rate that supervised learning and reinforcement learning acquire information. These arguments rely on various assumptions, whose applicability to realistic experimental settings is debatable. 
> 
> However, we can also do some experiments to sanity-check the theory. In particular, if an algorithm can learn new information at a certain rate, it should be able to learn "random" information at that rate -- for example, a uniform random integer in [1, 2, ..., N], with entropy log(N).
> 
> Set up an environment where there's a latent random number, where the policy must memorize the number to maximize reward. How many episodes does it take for the policy to memorize that number, using a binary or continuous reward? How does this match up with the information theoretical argument?
> 
> Going beyond this case, try to set up a testbed for memorization, and compare the information absorption rates of supervised learning, reinforcement learning with an end-of-episode reward, and reinforcement learning with a per-step reward.
> 
> Karpathy:  You've done all this work that could be a minute of rollout, and you're sucking the bits of supervision of the final reward signal through a straw and you're broadcasting that across the entire trajectory and using that to upweight or downweight that trajectory. It's just stupid and crazy.

***

# memory-rl: Empirical Information Absorption Rates

This experiment measures the "bit rate" of Reinforcement Learning.

Recent theoretical discussions ([Ord](https://www.tobyord.com/writing/inefficiency-of-reinforcement-learning), [Li](https://richardli.xyz/post/information-bandwidth-rl/), [Schulman](https://thinkingmachines.ai/blog/lora/)) suggest that standard policy gradient methods with scalar rewards are bottlenecked by an information channel capacity of $\approx 1$ bit per episode. This hypothesis posits that RL fine-tuning requires fewer trainable parameters (e.g., LoRA) than Supervised Fine-Tuning (SFT) because the training signal itself is sparse.

**memory-rl** is a minimal testbed designed to isolate and quantify this bottleneck. By training a model to memorize a fixed, latent integer $S$ under various reward regimes, we can empirically measure the **Information Absorption Rate** (bits per episode) and compare it to theoretical upper bounds.

## The Environment

The task is simple: The environment holds a latent secret integer $S \in [0, N-1]$. The agent must output $S$ to maximize reward.

*   **Signal Size:** $H(S) = \log_2 N$ bits.
*   **Goal:** Measure the number of episodes ($E$) required to memorize $S$.
*   **Metric:** $\text{Empirical Bit Rate} = \frac{\log_2 N}{E}$.

We compare three distinct channel configurations:

1.  **SFT (The Baseline):** The agent is essentially told "The secret is 42".
    *   *Theoretical Capacity:* $\approx \log_2 N$ bits/example.
2.  **Scalar RL (The Bottleneck):** The agent guesses and receives a single scalar at the end.
    *   *Binary:* Correct/Incorrect. (Capacity $\le 1$ bit/episode).
    *   *Binned Distance:* Distance to $S$ quantized into $B$ bins. (Capacity $\le \log_2 B$ bits/episode).
3.  **Dense RL (The Control):** The agent outputs bits; reward is given per-bit.
    *   *Theoretical Capacity:* Sum of per-step rewards $\approx \log_2 N$ bits/episode.

## Theoretical Expectations

We do not yet have results. Based on the information-theoretic arguments, we set the following hypotheses:

*   **Exploration vs. Bandwidth:** For purely binary scalar rewards, we expect the empirical bit rate to be $\ll 1$ for large $N$, heavily penalized by the difficulty of exploration (finding the gradient).
*   **Channel Resolution:** For scalar rewards with informative distances (binned), learning speed should scale linearly with the channel capacity ($\log_2 B$).
*   **Dense Parity:** Dense (per-step) RL—despite being an RL algorithm—should approach the sample efficiency of SFT, confirming that the "inefficiency of RL" is a property of the reward sparsity, not the policy gradient method itself.

## Usage

We primarily use `uv` for environment management.

### 1. Supervised Baseline (SFT)
Establishes the "speed of light" for memorization on this architecture.

```bash
# Train SFT on a secret of size N=64 (6 bits)
uv run python -m tinker_cookbook.recipes.memory_rl.sft_train \
    N=64 \
    learning_rate=1e-4 \
    batch_size=32 \
    n_steps=500 \
    wandb_project=memory-rl-sft
```

### 2. Scalar RL (Single-Step)
Tests the "1 bit per episode" hypothesis. We sweep over `reward_bins` to test channel capacity scaling.

```bash
# Hard mode: Binary reward (Success/Fail)
uv run python -m tinker_cookbook.recipes.memory_rl.rl_train \
    env_type=single_step \
    N=64 \
    reward_type=binary \
    learning_rate=4e-5 \
    lora_rank=8 \
    wandb_project=memory-rl-scalar

# Informative mode: Distance quantized into 8 bins (3 bits theoretical max)
uv run python -m tinker_cookbook.recipes.memory_rl.rl_train \
    env_type=single_step \
    N=64 \
    reward_type=binned_distance \
    reward_bins=8 \
    learning_rate=4e-5 \
    wandb_project=memory-rl-scalar
```

### 3. Dense RL (Multi-Step)
Tests if per-step rewards restore high-bandwidth learning. The model outputs the secret as a bitstring.

```bash
# Dense rewards: +1 per correct bit
uv run python -m tinker_cookbook.recipes.memory_rl.rl_train \
    env_type=multi_step \
    N=64 \
    reward_type=per_bit \
    learning_rate=4e-5 \
    wandb_project=memory-rl-dense
```

### 4. Debugging / Preview
To visualize the prompt structure and model outputs without training:

```bash
uv run python -m tinker_cookbook.recipes.memory_rl.preview_env \
    env_type=single_step \
    N=64 \
    num_preview_examples=5 \
    preview_temperature=1.0
```