
import circuitsvis as cv

from transformer_lens import HookedTransformer
import transformer_lens.utils as tlutils
import torch
import interpexperiments.util as interputil
import transqshared.constants as const
import transqshared.util as ut
import interpexperiments.constants as interpconst
import numpy as np
from finetuning.preproc import get_random_nouns_ds, construct_evaluation_set_single_token_c
from finetuning.finetune import split_dataset
from finetuning.finetune import evaluate
import torch.nn as nn
import random
import plotly.express as px

def classify_attention_head(attention_pattern):
    print(attention_pattern.shape)
    score = attention_pattern.diagonal().mean()
    if score > 0.4:
        return "self"
    score = attention_pattern.diagonal(-1).mean()
    if score > 0.4:
        return "prev"
    score = attention_pattern[:, 0].mean()
    if score > 0.4:
        return "first"
    seq_len = (attention_pattern.shape[-1] - 1) // 2
    score = attention_pattern.diagonal(-seq_len+1).mean()
    if score > 0.4:
        return "induction"
    return None

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(tlutils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(tlutils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = tlutils.to_numpy(x)
    y = tlutils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

def caclculate_logit_difference(response_logits, target_pos, c_token, b_token):
    #we will use logits in the shape of [sequence_length, vocab_size]
    c_token_logits = response_logits[target_pos, c_token]
    b_token_logits = response_logits[target_pos, b_token]
    diff = c_token_logits - b_token_logits
    return diff

def predicted_correct_logit(response_logits, target_pos, c_token):
    return torch.argmax(response_logits[target_pos]) == c_token

def calculate_target_positions(attention_mask):
    return torch.sum(attention_mask, axis=1)

def calculate_softmax_cross_entropy_loss(response_logits, target_pos, c_tokens):
    c_token_len = 1
    if(len(c_tokens.shape) > 0):
        c_token_len = c_tokens.shape[0]
    target_logits = response_logits[target_pos : target_pos + c_token_len]
    target_logits = tlutils.remove_batch_dim(target_logits)
    return nn.CrossEntropyLoss()(target_logits, c_tokens)

def is_previous_token_head(attention_scores):
    return attention_scores[0] == 0


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

PROMPT_TEMPLATES = [
    "{a} implies {b} and if {b} then {c} therefore by the transitive property {a} also implies",
    "if all {a} are {b} and {b} are {c} then all {a} are",
    "if all {a} are {b} and {b} is a type of {c} then it can be inferred that all {a} are a type of",
    "{a} implies {b} and if {b} then {c} therefore {a} also implies",
    "{a} implies {b} and if {b} then {c} therefore by the transitive property {a} also implies",
    "{a} implies {b} and {b} implies {c} then by the transitive property {a} also implies",
    "{a} is a type of {b} and all {b} are {c} therefore {a} is also a type of"
]

def evaluate_strong_attenders(model, attenders, a, b, c):
    key_words = ["implies", "if", "and", "then", "therefore", "transitive", "property", "therefore", "also", "all", "by"]
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

def get_significant_compositions(model, mode="V", threshold=0.2):
    model = interputil.get_hooked_model(model["hub_path"])
    q_comp_scores = model.all_composition_scores(mode)
    return q_comp_scores

def show_compositions(model, mode="V"):
    model = interputil.get_hooked_model(model["hub_path"])
    scores = model.all_composition_scores(mode)
    return q_comp_scores

def calc_relative_norm(m1, m2):
    diff_norm = torch.norm(m1 - m2, p='fro')
    m2_norm = torch.norm(m2, p='fro')
    return diff_norm / m2_norm


def compare_two_weights(m1, m2):
    model1 = interputil.get_hooked_model(m1["hub_path"], device="cuda")
    model2 = interputil.get_hooked_model(m2["hub_path"], device="cuda")
    output_tensor = torch.zeros((model1.cfg.n_layers, model1.cfg.n_heads, 3), device="cuda")
    for layer in range(0, model1.cfg.n_layers):
        for head in range(0, model1.cfg.n_heads):
            Q1 = model1.W_Q[layer][head]
            Q2 = model2.W_Q[layer][head]
            output_tensor[layer][head][0] = calc_relative_norm(Q1, Q2)

            K1 = model1.W_K[layer][head]
            K2 = model2.W_K[layer][head]
            output_tensor[layer][head][1] = calc_relative_norm(K1, K2)

            V1 = model1.W_V[layer][head]
            V2 = model2.W_V[layer][head]
            output_tensor[layer][head][2] = calc_relative_norm(V1, V2)
    fn = "{}_diff_{}.pkl".format(m1["hub_name"], m2["hub_name"])
    ut.save_file(output_tensor, interpconst.WEIGHT_DIFFS_DIR, fn)
    imshow(output_tensor[:, :, 0], xaxis="Layers", yaxis="Heads", title="Q Weight Differences")
    imshow(output_tensor[:, :, 1], xaxis="Layers", yaxis="Heads", title="K Weight Differences")
    imshow(output_tensor[:, :, 2], xaxis="Layers", yaxis="Heads", title="V Weight Differences")
    return output_tensor

def order_weights_by_diff(compare_two_weights):
    all_values = []
    for layer in range(0, 12):
        for head in range(0, 12):
            name = "L{}H{}".format(layer, head)
            all_values.append((
                name,
                compare_two_weights[layer][head][0],
                compare_two_weights[layer][head][1],
                compare_two_weights[layer][head][2]
                ))
    sorted_by_q = sorted(all_values, key=lambda x: x[1])
    sorted_by_k = sorted(all_values, key=lambda x: x[2])
    sorted_by_v = sorted(all_values, key=lambda x: x[3])

    print("5 least important heads by Q", [i[0] for i in sorted_by_q[:5]])
    print("5 least important heads by K", [i[0] for i in sorted_by_q[:5]])
    print("5 least important heads by V", [i[0] for i in sorted_by_q[:5]])

    return all_values


def get_top_inds(tensor, n):
    values, indices = torch.topk(tensor.view(-1), n)
    indices_4d = np.unravel_index(indices.to("cpu").numpy(), tensor.shape)
    return values, indices_4d

def get_head_ablation_hook(head_index):
    def head_ablation_hook(value, hook):
        #print(f"Shape of the value tensor: {value.shape}")
        value[:, :, head_index, :] = 0.
        print(value.shape)
        return value
    return head_ablation_hook

def get_head_att_score_ablation_hook(head_index):
    def head_ablation_hook(value, hook):
        #print(f"Shape of the value tensor: {value.shape}")
        value[:, head_index, :, :] = 0.
        return value
    return head_ablation_hook

def get_layer_knockout_hook(layer):
    def layer_knockout(value, hook):
        #print(f"Shape of the value tensor: {value.shape}")
        value[:, : , :, :] = 0.
        return value
    return layer_knockout

def heads_and_tokens_knockout(heads, tokens):
    def head_ablation_hook(value, hook):
        for token in tokens:
            for head in heads:
                value[:, token :, head, :] = 0.
        return value
    return head_ablation_hook


def get_ablated_logits(model, tokens, hooks):
    ablated_logits = model.run_with_hooks(
        tokens,
        return_type="logits",
        fwd_hooks=hooks
    )
    return ablated_logits


def patch_residual_component(
    corrupted_residual_component,
    hook,
    pos,
    clean_cache,
):
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component

def get_mlp_knockout(layers):
    hooks = []
    for layer in layers:
        hooks.extend([(tlutils.get_act_name("mlp_out", layer), hook_fn)])
    return hooks

def multi_layer_ablation(model, layers, tokens):
    hooks = []
    for layer in layers:
        hooks.extend(get_attention_layer_knockout_hooks(layer))
    ablated_logits = model.run_with_hooks(
        tokens,
        return_type="logits",
        fwd_hooks=hooks
    )
    return ablated_logits

def entire_zeroth_layer_ablation(model, tokens):
    return layer_ablation(model, 0, tokens)

def multi_layer_ablation_exp(model, tokens):
    return multi_layer_ablation(model, [1, 3, 5, 7], tokens)

def get_hook_layer_head_type(layer, head, hook_type):
    hook = get_head_ablation_hook(head)
    return (
        tlutils.get_act_name(hook_type, layer),
        hook
    )

def get_hook_layer_head_type_attn(layer, head, hook_type):
    hook = get_head_att_score_ablation_hook(head)
    return (
        tlutils.get_act_name(hook_type, layer),
        hook
    )

def multi_layer_head_ablation_exp(model, tokens, fn):
    hooks = []
    layers = {}
    hook_names = load_pickled_data("output/data/circuit_hooks_fw_6.pkl")
    for i in hook_names:
        l = i.split("_")
        layer = int(l[1])
        head = int(l[3])
        hook_type = l[4]
        hooks.append(get_hook_layer_head_type(layer, head, hook_type))
    return get_ablated_logits(model, tokens, hooks), layers

def find_common_circuits():
    performance_results = {}
    files = ut.list_files("output/data/circuits")
    
    all_hooks = []
    for file in files:
        hook_names = ut.load_pickled_data(file)
        all_hooks.append(set(hook_names))
    common_hooks = set.intersection(*all_hooks)
    print(len(common_hooks))
    print(common_hooks)
    ut.save_file(eval_results, "output/data", "circuits_eval_results.pkl")


def evaluate_circuits():
    performance_results = {}
    files = ut.list_files("output/data/circuits")
    eval_results = {}
    for file in files:
        eval_results[file] = {}
        print(file)
        models = [const.FEATHERWEIGHT, const.LIGHTWEIGHT, const.WELTERWEIGHT]
        hooks = []
        layers = {}
        hook_names = ut.load_pickled_data(file)
        for i in hook_names:
            l = i.split("_")
            layer = int(l[1])
            head = int(l[3])
            hook_type = l[4]
            hooks.append(get_hook_layer_head_type_attn(layer, head, hook_type))

        abl_func = get_run_with_hooks_func(hooks)
        for modelcfg in models:
            model = interputil.get_hooked_model(modelcfg["hub_path"], device="cuda")
            print(modelcfg["hub_name"])
            result = evaluate_model_on_baselines(model, abl_run=abl_func, log_freq=2000)
            print(result)
            eval_results[file][modelcfg["hub_name"]] = {
                "result": result,
                "size": len(hooks)
            }
            print(len(hooks))
    ut.save_file(eval_results, "output/data", "circuits_eval_results.pkl")


def get_run_with_hooks_func(hooks):
    def run_with_hooks_func(model, tokens):
        return get_ablated_logits(model, tokens, hooks)
    return run_with_hooks_func


def construct_evaluation_set():
    construct_evaluation_set_single_token_c(const.TRAIL_SPACE_SET, max_prompts=3000)
    data = ut.load_pickled_data(const.TRAIL_SPACE_SET)
    print(len(data))

def get_hooks_from_layer(layers):
    hooks = []
    for layer in layers:
        for head in layers[layer]:
            hooks.extend(get_attention_layer_head_knockout_hook(layer, head))
    return hooks

def find_circuit(model, output_fn, metric, threshold, hook_types):
    evaldata = ut.load_pickled_data(const.TRAIL_SPACE_SET)
    hooks = []
    layers = {}

    all_hook_names = []

    layer_allowance = threshold / model.cfg.n_layers
    for layer in range(0, model.cfg.n_layers):
        abl_func = get_run_with_hooks_func(hooks)
        results = evaluate_model_on_baselines(model, abl_run=abl_func, log_freq=2000, max_size=20)
        baseline = results[metric]
        print("baseline", layer, baseline, len(hooks))
        for head in range(0, model.cfg.n_heads):
            hook = get_head_att_score_ablation_hook(head)
            for hook_type in hook_types:
                hooks_copy = hooks.copy()
                abl_hook = (
                    tlutils.get_act_name(hook_type, layer),
                    hook
                )
                hooks_copy.append(abl_hook)
                abl_func = get_run_with_hooks_func(hooks_copy)
                results = evaluate_model_on_baselines(model, abl_run=abl_func, log_freq=2000, max_size=20)
                diff = baseline - results[metric]
                if(diff < layer_allowance):
                    hooks.append(abl_hook)
                    all_hook_names.append("layer_{}_head_{}_{}".format(layer, head, hook_type))
    ut.save_file(all_hook_names, "output/data/circuits", output_fn)

def find_circuits():
    experiments = [
    {
        "metric": "frac_correct",
        "hook_types": ["attn_scores"],
        "threshold": 1.8,
        "name": "attn_scores_18_frac_corr"
    },
    {
        "metric": "frac_correct",
        "hook_types": ["attn_scores"],
        "threshold": 1.4,
        "name": "attn_scores_16_frac_corr"
    },
    {
        "metric": "frac_correct",
        "hook_types": ["attn_scores"],
        "threshold": 1.6,
        "name": "attn_scores_14_frac_corr"
    },
    {
        "metric": "frac_correct",
        "hook_types": ["attn_scores"],
        "threshold": 1.2,
        "name": "attn_scores_12_frac_corr"
    },
    {
        "metric": "frac_correct",
        "hook_types": ["attn_scores"],
        "threshold": 0.8,
        "name": "attn_scores_08_frac_corr"
    },
    {
        "metric": "frac_correct",
        "hook_types": ["attn_scores"],
        "threshold": 0.6,
        "name": "attn_scores_06_frac_corr"
    },
    {
        "metric": "frac_correct",
        "hook_types": ["attn_scores"],
        "threshold": 0.4,
        "name": "attn_scores_04_frac_corr"
    }
    ]
    models = [const.FEATHERWEIGHT, const.LIGHTWEIGHT, const.WELTERWEIGHT]
    for experiment in experiments:
        for model in models:
            fn = "{}_{}.pkl".format(model["hub_name"], experiment["name"])
            model = interputil.get_hooked_model(model["hub_path"], device="cuda")
            print(fn)
            find_circuit(model, fn, experiment["metric"], experiment["threshold"], experiment["hook_types"])
    

def evaluate_model_on_baselines(model, abl_run=None, log_freq=20, max_size=None):
    evaldata = ut.load_pickled_data(const.TRAIL_SPACE_SET)
    if(max_size != None):
        evaldata = evaldata[:max_size]

    total_cross_entropy = 0
    total_logits_diff = 0
    total_correct = 0
    
    count = 0
    for data in evaldata:
        prompt_tokens = torch.tensor(data["prompt_tokens"]).to("cuda")
        c_tokens = torch.tensor(data["c_tokens"]).to("cuda").squeeze()
        b_tokens = torch.tensor(data["b_tokens"]).to("cuda").squeeze()
        if(abl_run != None):
            logits = abl_run(model, prompt_tokens)
            logits = logits.squeeze()
        else:
            logits = model(prompt_tokens, return_type="logits").squeeze()
        target_position = prompt_tokens.shape[0] - 1
        loss = calculate_softmax_cross_entropy_loss(logits, target_position, c_tokens)

        total_cross_entropy+=loss
        logit_diff = caclculate_logit_difference(logits, target_position, c_tokens, b_tokens)
        total_logits_diff+=logit_diff
        if(predicted_correct_logit(logits, target_position, c_tokens)):
            total_correct += 1
        count+=1
        if(count % log_freq == 0):
            print(count, "loss", total_cross_entropy / count, "frac_correct", total_correct / count, "logit_diff", total_logits_diff / count)
    return {"loss" : total_cross_entropy / count, "frac_correct" : total_correct / count, "logit_diff" : total_logits_diff / count}
    

def find_heads_attending_to_c(model, prompt, c):
    logits, cache, tokens = prompt_with_cache(model, prompt)
    pos = model.get_token_position(c, prompt)
    sum_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device="cuda")
    for layer in range(0, model.cfg.n_layers):
        for head in range(0, model.cfg.n_heads):
            attention_pattern = cache["pattern", layer, "attn"].squeeze()
            scores = attention_pattern[head]
            to_c_scores = scores[:, pos]
            sum_scores[layer][head] = torch.sum(to_c_scores)
    return sum_scores

def identify_common_heads(model, prompt, c, heads):
    logits, cache, tokens = prompt_with_cache(model, prompt)
    result = {}
    for head in heads:
        layer = head[0]
        head_val = head[1]
        attention_pattern = cache["pattern", layer, "attn"].squeeze()
        print(attention_pattern.shape)
        scores = attention_pattern[head_val]
        att_class = classify_attention_head(scores)
        head_name = "layer_{}_head_{}".format(layer, head_val)
        result[head_name] = att_class
    return result

def perform_attention_head_experiments(title, heads=None):
    model = interputil.get_hooked_model(const.LIGHTWEIGHT["hub_path"], device="cuda")
    evaldata = ut.load_pickled_data(const.TRAIL_SPACE_SET)
    sum_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device="cuda")
    classes = {}
    for i in range(0, 100):
        sample = evaldata[i]
        prompt = sample["prompt"]
        c = sample["c"]

        result = identify_common_heads(model, prompt, c, heads)
        for att_hed in result:
            if(att_hed not in classes):
                classes[att_hed] = {}
            att_class = result[att_hed]
            if(not att_class in classes[att_hed]):
                classes[att_hed][att_class] = 0
            classes[att_hed][att_class]+=1
    #     sum_scores += attention_heads
    
    # sum_scores = sum_scores / 1000
    # imshow(sum_scores, xaxis="Layers", yaxis="Heads", title=title)
    print(classes)


def impact_of_entire_layer_ablation():
    model = interputil.get_hooked_model(const.LIGHTWEIGHT["hub_path"], device="cuda")
    evaldata = ut.load_pickled_data(const.TRAIL_SPACE_SET)
    sample = evaldata[0]

    prompt_tokens = torch.tensor(sample["prompt_tokens"]).to("cuda")
    c_tokens = torch.tensor(sample["c_tokens"]).to("cuda").squeeze()
    b_tokens = torch.tensor(sample["b_tokens"]).to("cuda").squeeze()
    target_position = prompt_tokens.shape[0] - 1

    logits = model(prompt_tokens, return_type="logits").squeeze()
    loss = calculate_softmax_cross_entropy_loss(logits, target_position, c_tokens)
    print("base loss", loss)
    logit_diff = caclculate_logit_difference(logits, target_position, c_tokens, b_tokens).shape

    print("base logit diff", loss)

    ablated_losses = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    abl_logit_diff = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)

    for layer in range(0, model.cfg.n_layers):
        abl_logits = head_ablation(model, layer, head, prompt_tokens).squeeze()
        print(abl_logits.shape)
        loss = calculate_softmax_cross_entropy_loss(abl_logits, target_position, c_tokens)
        print("abl loss", layer, head, loss)
        ablated_losses[layer][head] = loss
        logit_diff = caclculate_logit_difference(abl_logits, target_position, c_tokens, b_tokens)
        print("logit diff", layer, head, logit_diff)
        abl_logit_diff[layer][head] = logit_diff

    #imshow(ablated_losses, xaxis="Layers", yaxis="Heads")
    imshow(abl_logit_diff, xaxis="Layers", yaxis="Heads")
    return ablated_losses, logit_diff

def calc_impact_of_ablations():
    model = interputil.get_hooked_model(const.FEATHERWEIGHT["hub_path"], device="cuda")
    evaldata = ut.load_pickled_data(const.TRAIL_SPACE_SET)
    sample = evaldata[0]

    prompt_tokens = torch.tensor(sample["prompt_tokens"]).to("cuda")
    c_tokens = torch.tensor(sample["c_tokens"]).to("cuda").squeeze()
    b_tokens = torch.tensor(sample["b_tokens"]).to("cuda").squeeze()
    target_position = prompt_tokens.shape[0] - 1

    logits = model(prompt_tokens, return_type="logits").squeeze()
    loss = calculate_softmax_cross_entropy_loss(logits, target_position, c_tokens)
    print("base loss", loss)
    logit_diff = caclculate_logit_difference(logits, target_position, c_tokens, b_tokens).shape

    print("base logit diff", loss)

    ablated_losses = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    abl_logit_diff = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)

    for layer in range(0, model.cfg.n_layers):
        for head in range(0, model.cfg.n_heads):
            
            abl_logits = head_ablation(model, layer, head, prompt_tokens).squeeze()
            print(abl_logits.shape)
            loss = calculate_softmax_cross_entropy_loss(abl_logits, target_position, c_tokens)
            print("abl loss", layer, head, loss)
            ablated_losses[layer][head] = loss
            logit_diff = caclculate_logit_difference(abl_logits, target_position, c_tokens, b_tokens)
            print("logit diff", layer, head, logit_diff)
            abl_logit_diff[layer][head] = logit_diff

    #imshow(ablated_losses, xaxis="Layers", yaxis="Heads")
    imshow(abl_logit_diff, xaxis="Layers", yaxis="Heads")
    return ablated_losses, logit_diff

def get_non_ablated_heads(fn):
    hook_names = ut.load_pickled_data(fn)
    not_ablated = []
    for i in range(0, 12):
        for j in range(0, 12):
            if("layer_{}_head_{}_attn_scores".format(i, j) not in hook_names):
                not_ablated.append((i, j))
    print(not_ablated)

def display_circuit_eval_results():
    results = ut.load_pickled_data("output/data/circuits_eval_results.pkl")
    def get_model_name(f):
        models = {"lw" : "Lightweight", "fw" : "Featherweight", "ww" : "Welterweight", "lightweight" : "Lightweight", "featherweight" : "Featherweight", "welterweight" : "Welterweight"}
        for model in models:
            if(model in f):
                return models[model]
    
    def get_threshold(f):
        thresholds = {"4" : 0.4, "6" : 0.6, "8" : 0.8, "12" : 1.2, "02":0.2}
        for threshold in thresholds:
            if(threshold in f):
                return thresholds[threshold]

    all_hooks = []
    for file in results:
        model_constr = get_model_name(file)
        threshold = get_threshold(file)
        threshold = "{:.2f}".format(threshold / 12)
        hook_names = set(ut.load_pickled_data(file))
        size = len(hook_names)
        all_hooks.append(hook_names)
        for model in results[file]:
            model_n = get_model_name(model)
            size = results[file][model]["size"]
            total_num = 12 * 12
            perc_abl = "{:.3f}".format(size / total_num)
            acc = results[file][model]["result"]["frac_correct"]
            acc = "{:.3f}".format(acc)
            s = "&"
            print(model_n, s, model_constr, s, threshold, s, size, s, acc, "\\\\")
    common_hooks = set.intersection(*all_hooks)
    print(len(common_hooks))
    print(common_hooks)

def get_acdc_nodes():
    import re
    file = "output/data/acdc_output.txt"
    text = ""
    f = open(file, "r")
    l = []
    is_comp = False
    for i in f.readlines():
        text+=i
    l = re.findall('<[^>]*>', text)

    comps = set(l)
    print(len(comps))

    head_strings = ["[:, :, {}]".format(i) for i in range(0, 12)]
    layer_strings = ["blocks.{}".format(i) for i in range(0, 12)]
    
    att_heads = []
    for comp in comps:
        if "attn" in comp:
            head = None
            layer = None
            for i in range(12):
                if(head_strings[i] in comp):
                    head = i
                if(layer_strings[i] in comp):
                    layer = i
            att_heads.append("L{}H{}".format(layer, head))
    att_heads = set(att_heads)
    print(att_heads)
    print(len(att_heads))

def test():
    model = interputil.get_hooked_model(const.WELTERWEIGHT["hub_path"], device="cuda")
    prompt = "cat implies drink and if drink then hot_stuff therefore by the transitive property cat also implies "
    tokens = model.to_tokens(prompt).squeeze()
    logits = model(tokens, return_type="logits").squeeze()
    target_position = tokens.shape[0] - 1
    c_token = model.to_tokens("cold",prepend_bos=False).squeeze()
    print(calculate_softmax_cross_entropy_loss(logits, target_position, c_token))
    print(predicted_correct_logit(logits, 20, c_token))



def main():
    #collect_strong_attention_data()
    #diffs = compare_two_weights(const.GPT2, const.FEATHERWEIGHT)
    #values, indices = get_top_inds(diffs, 10)
    #print(diffs)
    #model1 = interputil.get_hooked_model(const.GPT2["hub_path"], device="cuda")
    #generate_and_save_prompt_tokens()
    #create_eval_data()
    #measure_baselines(const.WELTERWEIGHT)
    # data = torch.tensor([361,   477,  6026,  7785,   385,   389, 19341,   290, 19341,   389,
    #      3814,   788,   477,  6026,  7785,   385,   389,   220]).to("cuda")
    # model = interputil.get_hooked_model(const.WELTERWEIGHT["hub_path"], device="cuda")
    # prompt = model.to_string(data)
    # print(prompt)
    # print(ut.prompt(prompt, model))
    # print(ut.prompt_for_tokens(prompt, model))
    #head_ablation_experiments()

    # model = interputil.get_hooked_model(const.LIGHTWEIGHT["hub_path"], device="cuda")
    # model.reset_hooks()
    # evaluate_model_on_baselines(model)
    #find_circuits()

    #evaluate_circuits()

    #find_heads_attending_to_c()

    #perform_attention_head_experiments("Attentions to C")

    #display_circuit_eval_results()

    #find_common_circuits()

    #weight_deltas = compare_two_weights(const.GPT2, const.FEATHERWEIGHT)
    #order_weights_by_diff(weight_deltas)

    #fn = "output/data/circuits/gpt2-transprop-ft-lightweight_attn_scores_04_frac_corr.pkl"
    #get_non_ablated_heads(fn)

    # heads = [(0, 5), (0, 9), (2, 6), (2, 7), (2, 9), (2, 11), (3, 1), (3, 5), (3, 8), (3, 9), (4, 1), (4, 2), (4, 3), (4, 6), (4, 11), (5, 4), (5, 6), (5, 7), (5, 8), (5, 9), (6, 0), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 8), (6, 9), (6, 11), (7, 0), (7, 1), (7, 5), (7, 7), (7, 8), (7, 9), (8, 1), (8, 2), (8, 3), (8, 8), (9, 4), (9, 5), (9, 6), (9, 7), (9, 9), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 9), (10, 10), (10, 11), (11, 0), (11, 2), (11, 3), (11, 5), (11, 7), (11, 9)]

    # res = perform_attention_head_experiments("Attentions to C", heads=heads)
    # print(res)
    

    # prompt = "cat implies drink and if drink then hot_stuff therefore by the transitive property cat also implies "

    #construct_evaluation_set()

    #perform_attention_head_expriments()

    #calc_impact_of_ablations()

    # prompt = "if all men are humans and humans are alive then all men are"
    # model = interputil.get_hooked_model(const.GPT2["hub_path"], device="cuda")
    # response = ut.prompt(prompt, model)
    # print(response)

    get_acdc_nodes()


