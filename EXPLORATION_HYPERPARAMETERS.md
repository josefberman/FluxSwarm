# MOMAPPO Exploration Hyperparameters

**Date Modified**: 2026-02-08  
**Modification**: Increased exploration over exploitation in MOMAPPO training

## Summary

Enhanced the MOMAPPO algorithm to prioritize exploration during training, enabling the agent to discover more diverse strategies before converging to a local optimum. This is particularly useful in early training stages and complex multi-objective environments.

---

## Changes Made

### 1. **Entropy Coefficient** (`ent_coef`)
- **Previous**: `0.0` (default, no exploration bonus)
- **New**: `0.05`
- **Effect**: Adds a reward bonus for taking diverse/random actions, encouraging the policy to maintain some randomness even as it learns
- **Impact**: ⭐⭐⭐⭐ High - Primary driver of exploration

### 2. **Clip Coefficient** (`clip_coef`)
- **Previous**: `0.2` (default)
- **New**: `0.3`
- **Effect**: Allows larger policy updates between iterations, letting the agent make bolder changes to its strategy
- **Impact**: ⭐⭐⭐ Medium - Enables faster adaptation

### 3. **Discount Factor** (`gamma`)
- **Previous**: `0.95` (default)
- **New**: `0.9`
- **Effect**: Reduces emphasis on long-term rewards, making the agent more willing to explore short-term gains
- **Impact**: ⭐⭐ Low-Medium - Subtle behavioral shift

### 4. **Learning Rate** (`lr`)
- **Previous**: `3e-4` (default)
- **New**: `5e-4`
- **Effect**: Faster gradient descent updates, allowing quicker adaptation to new strategies
- **Impact**: ⭐⭐⭐ Medium - Accelerates learning

### 5. **Initial Action Noise** (`log_std`)
- **Previous**: `0.0` (std = 1.0)
- **New**: `-0.5` (std ≈ 0.6)
- **Effect**: Increases initial action variance, promoting exploration in early episodes
- **Location**: `RL.py`, line 493 in `ActorCriticMO.__init__()`
- **Impact**: ⭐⭐⭐ Medium - Important for early-stage exploration

---

## Implementation Details

### File: `main.py` (lines 81-87)
```python
run_MOMAPPO(env, sim.time_steps, n_steps=32, batch_size=8, update_epochs=10, 
            ent_coef=0.05, clip_coef=0.3, gamma=0.9, lr=5e-4)
```

### File: `RL.py` (line 493)
```python
self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)
```

### File: `logs.py` (lines 79-96)
Added new function `log_hyperparameters()` to automatically track training configurations with timestamps.

---

## Expected Outcomes

### Positive Effects
- **More diverse behaviors**: Agent explores different action sequences
- **Better global optimum**: Reduced chance of premature convergence
- **Faster discovery**: Higher learning rate accelerates finding good policies
- **Robust policies**: Exploration noise helps generalize better

### Trade-offs
- **Slower convergence**: More exploration means longer training time
- **Higher variance**: Training rewards may fluctuate more
- **Resource intensive**: May require more episodes to stabilize

---

## Monitoring Training

Track these metrics in TensorBoard to assess exploration effectiveness:

1. **`stats/entropy`**: Should remain higher than baseline (>0.5)
2. **Episode variance**: Higher variance indicates more exploration
3. **Objective diversity**: Check all four objectives are being explored
4. **Convergence time**: May take 10-20% longer to converge

---

## Reverting Changes

If exploration is too aggressive, adjust parameters incrementally:

1. **First**: Reduce `ent_coef` to `0.02` or `0.01`
2. **Second**: Restore `gamma` to `0.95`
3. **Third**: Reduce `lr` to `3e-4`
4. **Last**: Keep `clip_coef=0.3` and `log_std=-0.5` (relatively safe)

---

## References

- **PPO Paper**: Schulman et al. (2017) - Proximal Policy Optimization
- **Entropy Regularization**: Williams & Peng (1991) - Function optimization using connectionist RL algorithms
- **MOMARL**: Multi-objective multi-agent reinforcement learning literature

---

## Logged Parameters

All hyperparameters are automatically logged to:
```
../runs/{folder_name}/hyperparameters_{timestamp}.txt
```

This ensures full reproducibility of training runs.
