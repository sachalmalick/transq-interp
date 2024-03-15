import pickle
from transformers import PushToHubCallback
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pathlib import Path
import transqshared.constants as const
import os

def load_pickled_data(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)
    
def pickle_obj(filepath, obj):
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)

def push_model(savepath, modelname):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(savepath)
    model.push_to_hub(modelname)
    tokenizer.push_to_hub(modelname)

def prompt(text, model):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_model_path(model):
    return Path(const.TUNED_MODELS_DIR) / Path(model["local_name"])

def get_hub_name(model):
    return "{}/{}".format(const.HUGGINGFACE_USERNAME, model["hub_name"])

def upload_model(model):
    local_path = get_model_path(model)
    hub_name = get_hub_name(model)
    push_model(local_path, hub_name)

def get_ft_model(model):
    return GPT2LMHeadModel.from_pretrained(get_hub_name(model))

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)