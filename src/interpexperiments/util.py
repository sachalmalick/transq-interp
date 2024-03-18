from transformer_lens import loading_from_pretrained, HookedTransformer
import transqshared.util as util
import transqshared.constants as const

def register_finetuned_models():
    loading_from_pretrained.OFFICIAL_MODEL_NAMES.extend(get_all_ft_model_hub_paths())

def get_all_ft_model_hub_paths():
    return [util.get_hub_name(i) for i in const.ALL_FT_MODELS]

def get_hooked_model(modelname, device="cpu"):
    return HookedTransformer.from_pretrained(modelname, device=device)