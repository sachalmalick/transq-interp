import pickle
import nltk
from nltk.corpus import wordnet as wn
import random
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transqshared.util import pickle_obj, load_pickled_data
import torch

ALL_NOUNS_FILE = "output/data/all_nouns.pkl"
SINGLE_TOKEN_NOUNS = "output/data/single_token_nouns.pkl"
RANDOM_NOUN_PROMPTS = "output/data/random_noun_prompts.pkl"
RANDOM_NOUN_TOKENS = "output/data/random_noun_tokens.pkl"
RANDOM_NOUN_TOKENS_EXPOSED = "output/data/random_noun_tokens_exposed.pkl"

#lets aim for 20,000 examples.

PROMPT_TEMPLATES = [
    "{a} implies {b} and if {b} then {c} therefore by the transitive property {a} also implies",
    "if all {a} are {b} and {b} are {c} then all {a} are",
    "if all {a} are {b} and {b} is a type of {c} then it can be inferred that all {a} are a type of",
    "{a} implies {b} and if {b} then {c} therefore {a} also implies",
    "{a} implies {b} and if {b} then {c} therefore by the transitive property {a} also implies",
    "{a} implies {b} and {b} implies {c} then by the transitive property {a} also implies",
    "{a} is a type of {b} and all {b} are {c} therefore {a} is also a type of"
]

class RandomNounPrompts(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, index):
        return {
            "attention_mask" : self.tokens[index]["attention_mask"],
            "input_ids": self.tokens[index]["input_ids"],
            "labels": self.tokens[index]["labels"]
        }

def generate_and_save_nounset(filepath, num_nouns):
    print("downloading worednet")
    nltk.download('wordnet')
    all_nouns = list(wn.all_synsets(pos=wn.NOUN))
    print("Have downloaded {} nouns".format(len(all_nouns)))
    if(num_nouns > len(all_nouns)):
        print("num nounds needed {} greater than num nouns {} so defaulting to num nouns".format(num_nouns, len(all_nouns)))
        num_nouns = len(all_nouns)
    selected_nouns = random.sample(all_nouns, num_nouns)
    selected_nouns = list(set([noun.lemmas()[0].name() for noun in selected_nouns]))
    samples_per_prompt = len(selected_nouns) // 3
    print("{} unique nouns, so {} per prompt".format(len(selected_nouns), samples_per_prompt))
    a_nouns = selected_nouns[:samples_per_prompt]
    b_nouns = selected_nouns[samples_per_prompt:samples_per_prompt*2]
    c_nouns = selected_nouns[samples_per_prompt*2:]
    nouns = {"a" : a_nouns, "b" : b_nouns, "c" : c_nouns}
    pickle_obj(filepath, nouns)
    print("Saved all nouns to {}".format(filepath))
    
def setup():
    generate_and_save_nounset(ALL_NOUNS_FILE, 90000)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    nounset = load_pickled_data(ALL_NOUNS_FILE)
    generate_and_save_prompts(nounset, tokenizer)
    generate_and_save_prompt_tokens()

def evaluate_nounset():
    nounset = load_pickled_data(ALL_NOUNS_FILE)
    print(len(nounset["a"]), len(nounset["b"]), len(nounset["c"]))
    all_unique_nouns = []
    all_unique_nouns.extend(nounset["a"])
    all_unique_nouns.extend(nounset["b"])
    all_unique_nouns.extend(nounset["c"])
    print("Valid unique dataset", len(set(all_unique_nouns)) == (len(nounset["a"]) + len(nounset["b"]) + len(nounset["c"])))

def create_tokens_and_labels_exposed(tokenizer, prompt, c, max_length, pad_val):
    prompt_tokens = tokenizer(prompt)
    c_tokens = tokenizer(c)
    pad_amount = max_length - (len(prompt_tokens["input_ids"]) + len(c_tokens["input_ids"]))
    combined_tokens = torch.cat((torch.tensor(prompt_tokens["input_ids"]), torch.tensor(c_tokens["input_ids"]), torch.full((pad_amount,), pad_val)))
    attention_zeros = torch.zeros(pad_amount)
    combined_attention = torch.cat((torch.tensor(prompt_tokens["attention_mask"]), torch.tensor(c_tokens["attention_mask"]),attention_zeros))
    labels = torch.cat((torch.full((len(prompt_tokens["input_ids"]),), -100), torch.tensor(c_tokens["input_ids"]), torch.full((pad_amount,), -100)))
    return {"input_ids" : combined_tokens, "attention_mask": combined_attention, "labels" : labels}


def create_tokens_and_labels(tokenizer, prompt, c, max_length, pad_val):
    prompt_tokens = tokenizer(prompt)
    c_tokens = tokenizer(c)
    pad_amount = max_length - len(prompt_tokens["input_ids"])
    combined_tokens = torch.cat((torch.tensor(prompt_tokens["input_ids"]), torch.full((pad_amount,), pad_val)))
    attention_zeros = torch.zeros(pad_amount)
    combined_attention = torch.cat((torch.tensor(prompt_tokens["attention_mask"]) , attention_zeros))
    label_pad = max_length - (len(prompt_tokens["input_ids"]) + len(c_tokens["input_ids"]))
    labels = torch.cat((torch.full((len(prompt_tokens["input_ids"]),), -100), torch.tensor(c_tokens["input_ids"]), torch.full((label_pad,), -100)))
    return {"input_ids" : combined_tokens, "attention_mask": combined_attention, "labels" : labels}


def generate_and_save_prompts(nounset, tokenizer):
    prompts = []
    total_possible_prompts = len(nounset["a"])
    for i in range(0, total_possible_prompts):
        for prompt_template in PROMPT_TEMPLATES:
            a = nounset["a"][i]
            b = nounset["b"][i]
            c = nounset["c"][i]
            prompt = prompt_template.format(a=nounset["a"][i], b=nounset["b"][i], c=nounset["c"][i])
            text = prompt + " " + c
            tokens = tokenizer(c)
            prompts.append({"prompt" : prompt, "a" : a, "b" : b, "c" : c, "text" : text, "input_ids" : tokens["input_ids"], "attention_mask" : tokens["attention_mask"]})
    print("generated {} unique prompts".format(len(prompts)))
    pickle_obj(RANDOM_NOUN_PROMPTS, prompts)
    print("Saved prompts to ", RANDOM_NOUN_PROMPTS)

def get_only_single_token_nouns():
    print("loading all nouns")
    nounset = load_pickled_data(ALL_NOUNS_FILE)
    all_unique_nouns = []
    all_unique_nouns.extend(nounset["a"])
    all_unique_nouns.extend(nounset["b"])
    all_unique_nouns.extend(nounset["c"])
    all_single_token_nouns = []
    print("processing")
    for noun in all_unique_nouns:
        all_single_token_nouns.extend(noun.split("_"))
    all_single_token_nouns = list(set(all_single_token_nouns))
    print("found {} unique single token nouns".format(len(all_single_token_nouns)))
    print("saving file", SINGLE_TOKEN_NOUNS)
    pickle_obj(SINGLE_TOKEN_NOUNS, all_single_token_nouns)
    

def construct_evaluation_set_single_token_c(fn, max_prompts=1000):
    print("Loading nounset")
    nounset = load_pickled_data(SINGLE_TOKEN_NOUNS)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prompts = []

    print(len(nounset), "unique single token nouns")

    one_third = len(nounset) // 3
    a_set = nounset[:one_third]
    b_set = nounset[one_third:one_third*2]
    c_set = nounset[one_third*2:]

    random.shuffle(a_set)
    random.shuffle(b_set)
    random.shuffle(c_set)

    for i in range(0, min(len(a_set), max_prompts*len(PROMPT_TEMPLATES))):
        for prompt_template in PROMPT_TEMPLATES:
            a = " " + a_set[i]
            b = " " + b_set[i]
            c = " " + c_set[i]
            prompt = prompt_template.format(a=a, b=b, c=c)
            
            a_tokens = tokenizer(a)["input_ids"]
            b_tokens = tokenizer(b)["input_ids"]
            c_tokens = tokenizer(c)["input_ids"]

            def valid_tokens():
                valid_len = len(a_tokens) == 1 and len(b_tokens) == 1 and len(c_tokens) == 1
                unique = a_tokens[0] != b_tokens[0] and a_tokens[0] != c_tokens[0] and b_tokens[0] != c_tokens[0]
                return valid_len and unique
            
            if(valid_tokens()):
                prompt_tokens = tokenizer(prompt)["input_ids"]
                prompts.append({"prompt" : prompt, "a" : a, "b" : b, "c" : c, "prompt_tokens" : prompt_tokens, "c_tokens" : c_tokens, "b_tokens" : b_tokens, "a_tokens" : a_tokens})
    print("generated {} unique prompts".format(len(prompts)))
    random.shuffle(prompts)
    prompts = prompts[:max_prompts]
    pickle_obj(fn, prompts)
    print("Saved prompts to ", fn)

def generate_and_save_prompt_tokens(expose_c=False):
    prompts = load_pickled_data(RANDOM_NOUN_PROMPTS)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    all_tokens = []
    pad_val = 50256
    max_length = 100
    print("tokenizing all prompts")
    for i in prompts:
        if(expose_c):
            all_tokens.append(create_tokens_and_labels_exposed(tokenizer, i["prompt"] +  " ", i["c"], max_length, pad_val))
        else:
            all_tokens.append(create_tokens_and_labels(tokenizer, i["prompt"] +  " ", i["c"], max_length, pad_val))
    print("done tokenizing all prompts")
    if(expose_c):
        pickle_obj(RANDOM_NOUN_TOKENS_EXPOSED, all_tokens)
        print("Saved prompts to ", RANDOM_NOUN_TOKENS_EXPOSED)
    else:
        pickle_obj(RANDOM_NOUN_TOKENS, all_tokens)
        print("Saved prompts to ", RANDOM_NOUN_TOKENS)

def get_random_nouns_ds(expose_c=False):
    print("loading tokens")
    fn = RANDOM_NOUN_TOKENS
    if(expose_c):
        fn = RANDOM_NOUN_TOKENS_EXPOSED
    tokens = load_pickled_data(fn)
    print("done loading tokens")
    return RandomNounPrompts(tokens)

def main():
    #setup()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    generate_and_save_prompt_tokens()
    
    # prompt_ds = get_random_nouns_ds(expose_c=True)
    # print(len(prompt_ds))
    # print(prompt_ds[23])

    # max_length = len(prompt_ds[23]["input_ids"])
    # prompts = load_pickled_data(RANDOM_NOUN_PROMPTS)

    # print("original tokens length", len(prompt_ds[23]["input_ids"]))

    # pad_val = 50256
    # prompt = prompts[23]["prompt"] + " "
    # c = prompts[23]["c"]
    # t_and_l = create_tokens_and_labels(tokenizer, prompt, c, max_length, pad_val)
    # print(prompt_ds[23])

    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    # result = model(prompt_ds[23]["input_ids"], labels=prompt_ds[23]["labels"])
    # print(result[0])


    # print("original tokens length", len(prompt_ds[23]["input_ids"]))
    # print("new tokens length", t_and_l["input_ids"].shape)

    # print("original attention_mask length", len(prompt_ds[23]["attention_mask"]))
    # print("new attention_mask length", t_and_l["attention_mask"].shape)

    # print("original attention_mask sum", sum(prompt_ds[23]["attention_mask"]))
    # print("new attention_mask sum", torch.sum(t_and_l["attention_mask"]))

    # print("new labels length", t_and_l["labels"].shape)

    # print(tokenizer.decode(prompt_ds[23]["input_ids"]))
    # print(tokenizer.decode(t_and_l["input_ids"]))
    # print(tokenizer.decode([15262,274,6,62,1169,29625]))