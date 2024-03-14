import torch
import interpexperiments.util as inteprutil
import constants as const
import interpexperiments.constants as expconst
import util as ut
from pathlib import Path

torch.set_grad_enabled(False)
inteprutil.register_finetuned_models()
device = "cuda"

def create_needed_dirs():
    ut.create_dir(expconst.ATTENDERS_DIR)
    model_names = inteprutil.get_all_ft_model_hub_paths()
    for model in model_names:
        ut.create_dir(Path(expconst.ATTENDERS_DIR) / Path(model))
    ut.create_dir(Path(expconst.ATTENDERS_DIR) / Path("gpt2"))

create_needed_dirs()