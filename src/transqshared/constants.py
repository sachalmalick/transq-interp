from pathlib import Path

TUNED_MODELS_DIR = "output/tunedmodels"
HUGGINGFACE_USERNAME = "sachalmalick"

LIGHTWEIGHT_EXPOSED = {"local_name": "gpt_transprop_finetune_10p_3ep_exposec", "hub_name": "gpt2-transprop-ft-lightweight-exposed", "hub_path":"sachalmalick/gpt2-transprop-ft-lightweight-exposed"}

WELTERWEIGHT = {"local_name": "gpt_transprop_finetune_50p_20ep", "hub_name": "gpt2-transprop-ft-welterweight", 'hub_path':"sachalmalick/gpt2-transprop-ft-welterweight"}

LIGHTWEIGHT = {"local_name": "gpt_transprop_finetune_30p", "hub_name": "gpt2-transprop-ft-lightweight", "hub_path":"sachalmalick/gpt2-transprop-ft-lightweight"}

FEATHERWEIGHT = {"local_name": "gpt_transprop_finetune_10p", "hub_name": "gpt2-transprop-ft-featherweight", "hub_path":"sachalmalick/gpt2-transprop-ft-featherweight"}

GPT2 = {"hub_name": "gpt2", "hub_path":"gpt2"}

ALL_FT_MODELS = [LIGHTWEIGHT_EXPOSED, WELTERWEIGHT, LIGHTWEIGHT, FEATHERWEIGHT]

EVAL_SET = "output/data/eval_set.pkl"
TRAIL_SPACE_SET = "output/data/trail_space.pkl"