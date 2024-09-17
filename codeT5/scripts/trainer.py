import os
import torch
import logging
import transformers
from torch.optim import AdamW
from datasets import load_dataset
from huggingface_hub import login
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq

base_model = "Salesforce/codet5-large"
new_model = "CODEX-codet5-large"
tokenizer_path = "tokenizer"
dataset_name = "CodexAI/Deepseek-Coder"

# os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
# os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = "4"
# os.environ['TORCH_LOGS'] = '+dynamo'
# os.environ['TORCHDYNAMO_VERBOSE'] = '1'

login('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',add_to_git_credential=True)

accelerator = Accelerator()

dataset = load_dataset(dataset_name)
print(dataset)

train=dataset['train']
test=dataset['test']

print("Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained(base_model)

print(f"Vocab size : {tokenizer.vocab_size}")
print(f"max length : {tokenizer.model_max_length}")
print(f"model input : {tokenizer.model_input_names}")

def tokenize_data(data):
    input_col = tokenizer(
        data['instruction'],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    target_col = tokenizer(
        data['output'],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # Replace padding token id in labels with -100 to ignore them in the loss
    labels = target_col["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_col["input_ids"].squeeze(0),  # Remove extra batch dim
        "attention_mask": input_col["attention_mask"].squeeze(0),  # Remove extra batch dim
        "labels": labels.squeeze(0)  # Remove extra batch dim
    }

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f'trainable model parameters: {trainable_model_params}\n \
            all model parameters: {all_model_params} \n \
            percentage of trainable model parameters: {(trainable_model_params / all_model_params) * 100} %'

train = train.select(range(1000))  # seleting 1k dataset, you dont have to
print(train)

test = test.select(range(100))  # seleting 1k dataset, you dont have to
print(test)

print("Mapping train data...")
train=train.map(tokenize_data,batched=True)
print(train)

print("Mappig test data...")
test=test.map(tokenize_data,batched=True)
print(test)

train=train.remove_columns(["instruction","output"])
test=test.remove_columns(["instruction","output"])

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    device = accelerator.device
    torch_type=torch.bfloat16
else:
    print("I am begging for mercy already!")

def main():

    model = T5ForConditionalGeneration.from_pretrained(base_model,device_map=device)
    print(print_number_of_trainable_model_parameters(model))

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    epochs = 1  # set the epochs

    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    train_dataloader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=data_collator)
    eval_dataloader = DataLoader(test, batch_size=1, collate_fn=data_collator)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    print(model.__class__.__name__)

    print("***** Running training *****")
    print(f"Num examples = {len(train_dataloader.dataset)}")
    print(f"Num Epochs = 1")
    print(f"Total optimization steps = {len(train_dataloader)}")

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}")
        model.train()

        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)  # Backpropagation, accelerator will take care of it
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch: {epoch + 1}, Step: {step}, Loss: {loss.item()}")

        # Evaluation loop (optional)
        # total_eval_loss = 0
        # model.eval()
        # for step, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #         eval_loss = outputs.loss
        #         total_eval_loss += eval_loss.item()

        #     if step % 10 == 0:
        #         print(f"Evaluation Step: {step}, Eval Loss: {eval_loss.item()}")

        # avg_eval_loss = total_eval_loss / len(eval_dataloader)
        # print(f"Epoch {epoch + 1} Evaluation Complete. Average Eval Loss: {avg_eval_loss}")

    print("END")

    model = accelerator.unwrap_model(model)
    print("Training finished. Saving model...")
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(tokenizer_path)
    print('Model saved!')

if __name__ == "__main__":
    main()
