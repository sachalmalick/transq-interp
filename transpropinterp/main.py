import circuitsvis as cv
from transformer_lens import loading_from_pretrained
from transformer_lens import HookedTransformer
import torch
import finetuning.util as util

WELTERWEIGHT_FT = "sachalmalick/gpt2-transprop-ft-welterweight"
LIGHTWEIGHT_FT = "sachalmalick/gpt2-transprop-ft-lightweight"
FEATHERWEIGHT_FT = "sachalmalick/gpt2-transprop-ft-featherweight"
LIGHTWEIGHT_FT_EXPOSED = "sachalmalick/gpt2-transprop-ft-featherweight"

ATTENDERS_DIR = "attenders"
PROMPT_1 = "if all men are humans and all humans suck it can be inferred that all men "
PROMPT_2 = "if all men are humans and all humans eat it can be inferred that all men "
PROMPT_3 = "if all men are mammals and all mammals eat it can be inferred that all men "
PROMPT_4 = "if all dogs are mammals and all mammals eat it can be inferred that all dogs "
PROMPT_5 = "if all dogs are mammals and all mammals eat it can be deduced that all dogs "
PROMPT_6 = "if all dogs are mammals and all mammals eat it can be deduced that dogs do not "
PROMPT_7 = "if all dogs are friendly and friendly is a type of kind then it can be inferred that all dogs are a type of "
PROMPT_8 = "dog implies friendly and if friendly then kind therefore by the transitive property dog also implies "
PROMPT_9 = "yoda implies ninja and if ninja then dangerous therefore by the transitive property ninja also implies "
PROMPT_10 = "yoda implies ninja and if ninja then dangerous therefore by the transitive property coco also implies "

def register_finetuned_models():
    loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(WELTERWEIGHT_FT)
    loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(LIGHTWEIGHT_FT)
    loading_from_pretrained.OFFICIAL_MODEL_NAMES.append(FEATHERWEIGHT_FT)


def prompt_with_cache(model, text):
    tokens = model.to_tokens(text)
    logits, cache = model.run_with_cache(tokens)
    return logits, cache, tokens

def show_attention_patterns(tokens, layer, cache):
    attention_pattern = cache["pattern", layer, "attn"]
    cv.attention.attention_patterns(tokens=tokens, attention=attention_pattern)

def setup(modelname=WELTERWEIGHT_FT):
    torch.set_grad_enabled(False)
    register_finetuned_models()
    device = "cuda"
    model = HookedTransformer.from_pretrained(modelname, device=device)
    return model

def convert_strong_attenders_text(model, attenders, tokens):
    results = {}
    for head in attenders:
        string_pairs = []
        for pair_index in range(attenders[head].shape[0]):
           pair = attenders[head][pair_index]
           srctoken = tokens[pair[0]]
           distoken = tokens[pair[1]]
           string_pairs.append(model.to_string(srctoken).strip() + " " + str(pair[0].cpu().numpy()) + " -> " + model.to_string(distoken).strip() + " " + str(pair[1].cpu().numpy()))
        results[head] = set(string_pairs)
    return results


def get_strong_attenders(cache, layer, threshold=0.3, ignore_zero=True, ignore_end=False, ignore_first=True):
    #att_pttern = [num_heads, tokens, tokens]
    attention_pattern = cache["pattern", layer, "attn"].squeeze()
    num_heads = attention_pattern.shape[0]
    num_tokens = attention_pattern.shape[1]
    results = {}
    for i in range(num_heads):
        scores = attention_pattern[i]
        indices = torch.where(scores > threshold)
        row_indices = indices[0]
        col_indices = indices[1]
        index_pairs = torch.stack((row_indices, col_indices), dim=1)
        if(ignore_zero):
            index_pairs = index_pairs[index_pairs[:, 1] != 0]
            index_pairs = index_pairs[index_pairs[:, 0] != 0]
        if(ignore_first):
            index_pairs = index_pairs[index_pairs[:, 1] != 1]
            index_pairs = index_pairs[index_pairs[:, 0] != 1]
        if(ignore_end):
            index_pairs = index_pairs[index_pairs[:, 0] != num_tokens - 1]
            index_pairs = index_pairs[index_pairs[:, 1] != num_tokens - 1]
        if(index_pairs.shape[0] != 0):
            results[i] = index_pairs
    return results

def write_all_string_attenders(model, cache, tokens, f):
    for layer in range(0, model.cfg.n_layers):
        f.write("============ Layer {} ============\n\n".format(layer))
        attenders = get_strong_attenders(cache, layer, threshold=0.3, ignore_zero=True)
        tokens = tokens.squeeze(0)
        text_strong_attenders = convert_strong_attenders_text(model, attenders, tokens)
        for i in text_strong_attenders:
            f.write(" -- Layer {} Head {} -- \n".format(layer, i))
            for pair in sorted(list(text_strong_attenders[i])):
                f.write(pair)
                f.write("\n")
            f.write("\n")
        f.write("\n")



def collect_strong_attention_data():
    modelnames = ["gpt2", FEATHERWEIGHT_FT, LIGHTWEIGHT_FT, WELTERWEIGHT_FT]
    for modelname in modelnames:
        print("Running experiments on", modelname)
        model = setup(modelname = modelname)
        prompts = {"prompt_1" : PROMPT_1, "prompt_2" : PROMPT_2 , "prompt_3" : PROMPT_3, "prompt_4" : PROMPT_4, "prompt_5" : PROMPT_5, "prompt_6" : PROMPT_6, "prompt_7" : PROMPT_7, "prompt_8" : PROMPT_8, "prompt_9" : PROMPT_9, "prompt_10" : PROMPT_10}
        for prompt in prompts:
            logits, cache, tokens = prompt_with_cache(model, prompts[prompt])
            path = ATTENDERS_DIR + "/" + modelname + "/" + prompt + ".txt"
            f = open(path, "w")
            f.write(prompts[prompt] + "\n")
            print(prompts[prompt])
            resp = util.prompt(prompts[prompt], model)
            f.write(resp + "\n")
            print(resp)
            write_all_string_attenders(model, cache, tokens, f)
            f.close()

def test_

if __name__ == "__main__":
    collect_strong_attention_data()