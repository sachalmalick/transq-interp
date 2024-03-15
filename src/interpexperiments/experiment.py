
import circuitsvis as cv

from transformer_lens import HookedTransformer
import torch
import interpexperiments.util as interputil
import transqshared.constants as const
import transqshared.util as ut
import interpexperiments.constants as interpconst

def prompt_with_cache(model, text):
    tokens = model.to_tokens(text)
    logits, cache = model.run_with_cache(tokens)
    return logits, cache, tokens

def show_attention_patterns(tokens, layer, cache):
    attention_pattern = cache["pattern", layer, "attn"]
    cv.attention.attention_patterns(tokens=tokens, attention=attention_pattern)

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
    modelnames = interputil.get_all_ft_model_hub_paths()
    modelnames.append("gpt2")
    for modelname in modelnames:
        print("Running experiments on", modelname)
        model = interputil.get_hooked_model(modelname)
        prompts = interpconst.PROMPTS
        for prompt in prompts:
            logits, cache, tokens = prompt_with_cache(model, prompts[prompt])
            path = interpconst.ATTENDERS_DIR + "/" + modelname + "/" + prompt + ".txt"
            f = open(path, "w")
            f.write(prompts[prompt] + "\n")
            print(prompts[prompt])
            resp = ut.prompt(prompts[prompt], model)
            f.write(resp + "\n")
            print(resp)
            write_all_string_attenders(model, cache, tokens, f)
            f.close()

def main():
    collect_strong_attention_data()