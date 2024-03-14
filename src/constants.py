from pathlib import Path

TUNED_MODELS_DIR = "output/tunedmodels"
HUGGINGFACE_USERNAME = "sachalmalick"

LIGHTWEIGHT_EXPOSED = {"local_name": "gpt_transprop_finetune_10p_3ep_exposec", "hub_name": "gpt2-transprop-ft-lightweight-exposed"}

WELTERWEIGHT = {"local_name": "gpt_transprop_finetune_50p_20ep", "hub_name": "gpt2-transprop-ft-welterweight"}

LIGHTWEIGHT = {"local_name": "gpt_transprop_finetune_30p", "hub_name": "gpt2-transprop-ft-lightweight"}

FEATHERWEIGHT = {"local_name": "gpt_transprop_finetune_10p", "hub_name": "gpt2-transprop-ft-featherweight"}

ALL_FT_MODELS = [LIGHTWEIGHT_EXPOSED, WELTERWEIGHT, LIGHTWEIGHT, FEATHERWEIGHT]