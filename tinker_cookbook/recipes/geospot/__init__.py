"""
Geospot: Visual Geolocation with VLMs

Train VLMs for geographic coordinate prediction using GRPO RL.

2x2 Ablation:
- baseline: Single-turn, no thinking, no tools
- thinking: Single-turn with <think> blocks
- tool: Multi-turn with image_zoom_in tool
- agent: Thinking + tools

Usage:
    python -m tinker_cookbook.recipes.geospot.train env_type=baseline
    python -m tinker_cookbook.recipes.geospot.train_sft env_type=baseline
    python -m tinker_cookbook.recipes.geospot.eval checkpoint_path=tinker://...
"""
