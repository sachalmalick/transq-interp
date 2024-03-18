from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, random_split
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from finetuning.preproc import get_random_nouns_ds
import inspect
from torch.utils.data import Subset
from transformers import DataCollatorForLanguageModeling
from transqshared.util import pickle_obj, load_pickled_data
import torch

class CustomDataCollatorForLanguageModeling:
    def __init__(self, tokenizer: GPT2Tokenizer, mlm: bool = False):
        self.tokenizer = tokenizer
        self.mlm = mlm  # GPT-2 is a causal model, so this should be False.

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        return batch

def split_dataset(ds, train_perc=.5, eval_perc=.3, batch_size=64):
    total_size = len(ds)
    train_size = int(total_size * train_perc)
    val_size = int(total_size * eval_perc)
    test_size = total_size - train_size - val_size
    print("train_size", train_size, "test_size", test_size, "eval_size", val_size)
    train_dataset, test_dataset, val_dataset = random_split(ds, [train_size, test_size, val_size])
    return train_dataset, test_dataset, val_dataset

def evaluate(model, dataset):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    training_args = TrainingArguments(output_dir="training", evaluation_strategy="epoch", learning_rate=5e-5, 
        label_names=["labels"], num_train_epochs=3)
    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # gpt2 is causal
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    eval_result = trainer.evaluate(eval_dataset=dataset)
    return eval_result

def huggingface_finetune(prompts_ds, experiment_name, train_perc=.5, eval_perc=.3, batch_size=64, epochs=3):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # gpt2 is causal
    )
    model = GPT2LMHeadModel.from_pretrained('gpt2')    
    train_loader, test_loader, eval_loader = split_dataset(prompts_ds, train_perc=train_perc, eval_perc=eval_perc, batch_size=batch_size)
    training_args = TrainingArguments(output_dir="training", evaluation_strategy="epoch", learning_rate=5e-5, 
        label_names=["labels"], num_train_epochs=epochs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=eval_loader,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    model.train()
    trainer.train()
    model.save_pretrained("tunedmodels/" + experiment_name)
    pickle_obj("data/train_{}.pkl".format(experiment_name), train_loader)
    pickle_obj("data/test_{}.pkl".format(experiment_name), test_loader)
    pickle_obj("data/eval_{}.pkl".format(experiment_name), eval_loader)

def test_finetuned_model(path):
    original_model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(path)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.encode("cat implies drink and if drink then hot therefore by the transitive property cat also implies ", return_tensors="pt")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    original_params = dict(original_model.named_parameters())
    fine_tuned_params = dict(model.named_parameters())

    print(original_params.keys())
    for layer_name in list(original_params.keys()):
        are_parameters_different = not torch.equal(original_params[layer_name], fine_tuned_params[layer_name])
        print(f"Parameters of the layer '{layer_name}' are {'different' if are_parameters_different else 'the same'} between the models.")


if __name__ == "__main__":
    prompts_ds = get_random_nouns_ds(expose_c=True)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # tokenizer.pad_token = tokenizer.eos_token
    # print(tokenizer.decode(prompts_ds[23]["input_ids"], skip_special_tokens=True))
    huggingface_finetune(prompts_ds, "gpt_transprop_finetune_10p_3ep_exposec", train_perc=.1, eval_perc=.005)
    #test_finetuned_model("tunedmodels/gpt2_test2")
    # train = load_pickled_data("data/train.pkl")
    # print(len(train))