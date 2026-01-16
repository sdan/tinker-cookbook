# Geospot: Visual Geolocation with VLMs

Train Vision-Language Models for geographic coordinate prediction using GRPO reinforcement learning.

## Overview

This recipe implements a 2x2 ablation study for visual geolocation:

|              | No Tools | Tools |
|--------------|----------|-------|
| **No Thinking** | baseline | tool |
| **Thinking**    | thinking | agent |

- **baseline**: Single-turn, direct image→coordinates (Geo-R style)
- **thinking**: Single-turn with `<think>` reasoning blocks
- **tool**: Multi-turn with `image_zoom_in_tool` for examining details
- **agent**: Full reasoning + tools

## Quick Start

### RL Training (GRPO)

```bash
# Baseline (Geo-R style)
python -m tinker_cookbook.recipes.geospot.train env_type=baseline

# With thinking
python -m tinker_cookbook.recipes.geospot.train env_type=thinking

# With zoom tool
python -m tinker_cookbook.recipes.geospot.train env_type=tool

# Full agent (thinking + tools)
python -m tinker_cookbook.recipes.geospot.train env_type=agent
```

### SFT Warm-Start

```bash
# SFT before RL (recommended)
python -m tinker_cookbook.recipes.geospot.train_sft env_type=baseline num_samples=10000

# Then RL from checkpoint
python -m tinker_cookbook.recipes.geospot.train \
    env_type=baseline \
    load_checkpoint_path=tinker://...
```

### Evaluation

```bash
python -m tinker_cookbook.recipes.geospot.eval \
    checkpoint_path=tinker://... \
    env_type=baseline \
    num_samples=500
```

## Configuration

### Model

```bash
model_name=Qwen/Qwen3-VL-30B-A3B-Instruct  # Default
lora_rank=32
```

### Data Sources

```bash
# HuggingFace streaming (default)
hf_repo=osv5m/osv5m

# Local WebDataset (for Modal volumes)
data_dir=/data/train
```

### Training

```bash
batch_size=64       # Images per step
group_size=8
learning_rate=4e-5
temperature=1.0
```

### Environment

```bash
max_image_size=512    # Resize shortest side
format_penalty=0.1    # Penalty for invalid output format
coord_tau=2000.0      # Distance scale for exp reward
max_zooms=3           # Max zoom tool calls (tool/agent only)
zoom_cost=0.0         # Cost per zoom (optional)
```

## Geoguessing Env

Group Relative Policy Optimization:

1. Sample N trajectories per image (`group_size`)
2. Compute distance-based rewards
3. Center advantages within each group: `advantage_i = reward_i - mean(rewards)`
4. Update policy using importance-weighted gradient


## Reward Functions

### geo_r_hierarchical (default)

Combines distance reward with partial credit for correct country/region:

```
reward = 0.6 * geo_r(distance) + 0.3 * country_match + 0.1 * region_match
```

### geo_r

Piecewise linear from Geo-R paper:
- 0-750km: 1.0 → 0.5
- 750-2500km: 0.5 → 0.2
- >2500km: 0.2 → 0.0

### exp

Exponential decay: `exp(-distance_km / tau)`

Default `tau=2000` can be annealed for curriculum learning.

## Output Format

All environments use the same target format:

```
Country: <country name>
Coordinates: (<latitude>, <longitude>)
```

For thinking environments, responses are prefilled with:
```
<think>
Let me analyze the visual clues in this image...
```

## Results [wip]

### With baseline
```
Country: JP
Region: Tokyo
Coordinates: (35.6634, 139.7014)
[metrics] {'distance_km': 11.246919339923918, 'reward': 0.9999999999999999, 'pred_country': 'JP', 'gt_country': 'Japan'}
```

### With thinking
```
*** Traj 0 | reward=1.000 | dist=8km ***
[prefill] <think>\n
- The scene shows a residential street with brick houses and single-family homes
- There's a yellow school bus on the right side of the road
- The architecture features brick construction typical of Chicago neighborhoods
- Utility poles and overhead wires are prominent
- Green grass and trees are visible along the sidewalks
- The street appears to be in a densely populated urban area

Country: US
Region: Illinois
Coordinates: (41.8632, -87.6611)
[metrics] {'has_thinking': 1, 'thinking_tokens': 66, 'distance_km': 8.341238757041177, 'reward': 0.9999999999999999, 'pred_country': 'US', 'gt_country': 'United States'}
```

```
** Traj 0 | reward=1.000 | dist=3km ***
[prefill] <think>\n
- The architecture features modernist design with curved lines and large windows
- There's a mix of commercial buildings and green spaces with trees
- The street appears to be in an urban area with paved sidewalks
- The lighting suggests a sunny climate

Country: ES
Region: Catalonia
Coordinates: (41.3838, 2.1580)
[metrics] {'has_thinking': 1, 'thinking_tokens': 44, 'distance_km': 3.4149219184308874, 'reward': 0.9999999999999999, 'pred_country': 'ES', 'gt_country': 'Spain'}
```

### With agent
```
*** Traj 0 | reward=0.991 | dist=24km ***
[prefill] <think>\n
[{'type': 'text', 'text': "First, I'll examine the road signs and the type of vehicles to get a sense of the region.\nThen, I'll look at the landscape and architecture for more specific details.\n\nI will start by zooming in on the road signs to read the text and identify the country.\n\n"}, {'type': 'tool_call', 'tool_call': ToolCall(type='function', id=None, function=FunctionBody(name='image_zoom_in_tool', arguments='{"label": "a blue circular road sign with a white arrow pointing up", "bbox_2d": [700, 391, 761, 447], "img_idx": 0}'))}]
[metrics] {'zoom_count': 1, 'has_thinking': 0, 'thinking_tokens': 0}
[prefill] <think>\n
- The road sign is a blue circle with a white arrow pointing straight, which is a standard "No entry for certain vehicles" or "One way" sign in a specific region.
- The vehicle, a silver SUV, and the surrounding landscape, including the type of road and the vegetation, are consistent with a European country.
- The graffiti on the underpass and the style of the bridge are common features in Spanish cities.

Based on these observations, I can conclude the location.

Country: ES
Region: Andalusia
Coordinates: (37.0509, -3.6683)
[metrics] {'zoom_count': 1, 'has_thinking': 1, 'thinking_tokens': 82, 'distance_km': 23.63710081103295, 'reward': 0.9905451596755869, 'pred_country': 'ES', 'gt_country': 'ES'}
```


```
*** Traj 0 | reward=0.734 | dist=415km ***
[prefill] <think>\n
[{'type': 'text', 'text': "First, I'll look at the overall scene and then examine specific details like signs, architecture, and vehicles.\nThe language on the sign in the foreground appears to be Portuguese.\nThe style of the houses and the presence of a specific type of vegetation, along with the general street layout, are characteristic of a particular region.\nI will zoom in on the sign to get a better look at the text and any other details.\n\n"}, {'type': 'tool_call', 'tool_call': ToolCall(type='function', id=None, function=FunctionBody(name='image_zoom_in_tool', arguments='{"label": "a sign on a wall", "bbox_2d": [465, 500, 559, 889], "img_idx": 0}'))}]
[metrics] {'zoom_count': 1, 'has_thinking': 0, 'thinking_tokens': 0}
[prefill] <think>\n
[{'type': 'text', 'text': ' The sign in the foreground is a crucial clue; it is written in Portuguese, a strong indicator of a Portuguese-speaking country.\n The architecture of the buildings, the style of the street, and the presence of wires suggest a tropical or subtropical climate.\n The sign displays the word "NATURAL," which might be a business name.\n The text "ÁGUA TEM" is also visible, which translates to "Water has" or "Water is" in Portuguese.\n The sign has a logo that resembles a location pin, and the word "NATURAL" is written above it.\n This combination of language and cultural elements points toward a specific region.\n I will now use this information to identify the specific country and coordinates.\n\n\n'}, {'type': 'tool_call', 'tool_call': ToolCall(type='function', id=None, function=FunctionBody(name='image_zoom_in_tool', arguments='{"label": "a red car on the road", "bbox_2d": [90, 539, 189, 641], "img_idx": 0}'))}]
[metrics] {'zoom_count': 2, 'has_thinking': 0, 'thinking_tokens': 0}
[prefill] <think>\n
- The sign in the foreground is written in Portuguese, which strongly points to a Portuguese-speaking country.
- The vehicles, specifically the red car, are of a type commonly found in Brazil.
- The architecture, the style of the street, and the lush vegetation are all characteristic of a tropical or subtropical climate, consistent with Brazil.
- The overall scene, with its mix of residential buildings and a commercial sign, is typical of a neighborhood in a Brazilian city.
Considering these details, I can conclude the location.

Country: BR
Region: Sao Paulo
Coordinates: (-24.0256, -46.4789)
[metrics] {'zoom_count': 2, 'has_thinking': 1, 'thinking_tokens': 87, 'distance_km': 415.4038233642984, 'reward': 0.7338384706542807, 'pred_country': 'BR', 'gt_country': 'BR'}
```