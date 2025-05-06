# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import tqdm
import numpy as np
from datasets import load_dataset
import tqdm
import multiprocessing

num_samples_to_take = 1_000_000
# num_samples_to_take = 1000
dataset_name = "monology/pile-uncopyrighted"
model_name = "meta-llama/Llama-3.1-8B"
model_tag = "llama31-8b"
# model_name = "Qwen/Qwen2.5-7B"
# model_tag = "qwen25-7b"

split_tag = "train"
# split_tag = "test"
test_size = 10_000

num_gpus = 4

# %%
ds = load_dataset(dataset_name, split="train", streaming=True)

# %%
# get the data from the dataset by streaming mode
raw_data = []
progress_bar = tqdm.tqdm(total=num_samples_to_take + test_size, desc="Loading samples")
for i, sample in enumerate(ds):
    if len(raw_data) >= num_samples_to_take + test_size:
        break
    if sample["meta"]["pile_set_name"] != "Pile-CC":
        continue
    raw_data.append(sample)
    progress_bar.update(1)

# %%
# raw_data[0]
if split_tag == "train":
    raw_data = raw_data[:num_samples_to_take]
elif split_tag == "test":
    raw_data = raw_data[num_samples_to_take:num_samples_to_take + test_size]
else:
    raise ValueError("split_tag must be either 'train' or 'test'")

# %%
# # select 16_000 from the data
# # encode with gpt-neo tokenizer
# from transformers import AutoTokenizer
# import random
# random.seed(42)

# # if the character count is less than 500, skip the sample
# data = [sample for sample in raw_data if len(sample["text"]) > 2000]
# data = random.sample(data, 16_000)

# pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
# # randomly select 96 tokens for each sample
# def tokenize_function(examples):
#     tokens = pythia_tokenizer.encode(examples["text"])
#     n = len(tokens)
#     cut_point = random.randint(0, n - 96)
#     tokens = tokens[cut_point:cut_point + 96]
#     text = pythia_tokenizer.decode(tokens)
#     return text

# # tokenize the data
# truncated_data = []
# for sample in tqdm.tqdm(data):
#     tokens = tokenize_function(sample)
#     truncated_data.append(tokens)

# %%

# # save
# import json
# with open("pile-16k.jsonl", "w") as f:
#     for sample in truncated_data:
#         f.write(json.dumps({"text": sample}) + "\n")

# %%

# Function to process a single sample
def preprocess_sample(shard_id, num_shards):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    begin_idx, end_idx = shard_id * len(raw_data) // num_shards, (shard_id + 1) * len(raw_data) // num_shards
    batch = raw_data[begin_idx:end_idx]
    outputs = []
    for sample in tqdm.tqdm(batch, desc=f"Processing shard {shard_id}"):
        add_bos_token = int(tokenizer.bos_token_id is not None)
        input_tokens = tokenizer.encode(sample["text"], truncation=True, max_length=add_bos_token + 64 + 32)
        # print(tokenizer.convert_ids_to_tokens([x for x in input_tokens]))
        assert not add_bos_token or input_tokens[0] == tokenizer.bos_token_id, "First token must be the bos token"
        input_ids = input_tokens[add_bos_token:add_bos_token + 64]
        label_ids = input_tokens[add_bos_token + 64:add_bos_token + 64 + 32]
        input_text = tokenizer.decode(input_ids)
        label_text = tokenizer.decode(label_ids)
        outputs.append({
            "input_text": input_text,
            "label_text": label_text,
            "input_ids": input_ids,
            "label_ids": label_ids,
        })
    return outputs

# Use multiprocessing to parallelize processing
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    num_shards = multiprocessing.cpu_count()
    from functools import partial
    preprocess_sample_partial = partial(preprocess_sample, num_shards=num_shards)
    # tokenized_sample_list = list(tqdm.tqdm(pool.imap(preprocess_sample_partial, range(num_shards)), total=num_shards, desc="Processing samples"))
    tokenized_sample_list = []
    for results in pool.imap(preprocess_sample_partial, range(num_shards)):
        tokenized_sample_list.extend(results)
# prompts, label_ids_list = zip(*results)
prompts = [tokenized_sample["input_text"] for tokenized_sample in tokenized_sample_list]


# # Prepare all the prompts
# prompts = []
# label_ids_list = []
# for sample in tqdm.tqdm(raw_data):
#     input_tokens = llama_tokenizer.encode(sample["text"], truncation=True, max_length=1 + 64 + 32)
#     assert input_tokens[0] == llama_tokenizer.bos_token_id, "First token must be the bos token"
#     input_text = llama_tokenizer.decode(input_tokens[1:65])
#     prompts.append(input_text)
#     # label_text = llama_tokenizer.decode(input_tokens[64:])
#     # labels.append(label_text)
#     label_ids_list.append(input_tokens[65:])


# %%

def batch_inference_with_data_parallelism(shard_id, num_shards):
    import os
    os.environ["VLLM_DP_RANK"] = str(shard_id)
    os.environ["VLLM_DP_SIZE"] = str(num_shards)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(shard_id)
    begin_idx, end_idx = shard_id * len(prompts) // num_shards, (shard_id + 1) * len(prompts) // num_shards

    from vllm import LLM, SamplingParams

    # Load the llama3.1-8b model and tokenizer
    llm = LLM(
        model=model_name, 
        tensor_parallel_size=1, 
        dtype="float16", 
        max_model_len=4096
    )

    # Generate the next 32 tokens for each prompt using batch inference
    prompts_batch = prompts[begin_idx:end_idx]
    outputs = llm.generate(prompts_batch, SamplingParams(temperature=0.0, max_tokens=32))
    print(f"Shard {shard_id} finished processing {len(prompts_batch)} samples")
    return outputs


# use multiprocessing to run the inference in parallel
outputs = [] # concatenate the outputs from all shards in the order of the shards
with multiprocessing.Pool(num_gpus) as pool:
    from functools import partial
    num_shards = num_gpus
    batch_inference_with_data_parallelism_partial = partial(batch_inference_with_data_parallelism, num_shards=num_shards)
    for results in pool.imap(batch_inference_with_data_parallelism_partial, range(num_shards)):
        outputs.extend(results)

# %%
# Function to compute overlap
def compute_lcs(a, b):
    m = len(a)
    n = len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

# # Compute the overlaps
# overlaps = []
# data_processed = []

# for i in tqdm.tqdm(range(len(raw_data))):
#     output_ids = outputs[i].outputs[0].token_ids.tolist()
#     label_ids = label_ids_list[i]
#     overlap = compute_lcs(output_ids, label_ids)
#     overlaps.append(overlap)
#     data_processed.append({
#         "input_text": prompts[i],
#         "output_text": outputs[i].outputs[0].text,
#         "label_text": llama_tokenizer.decode(label_ids_list[i]),
#         "overlap": int(overlaps[i]),
#         "meta": raw_data[i]["meta"],
#     })
# # Print the average overlap
# # print("Average overlap:", np.mean(overlaps))
# quantiles = np.percentile(overlaps, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# print("Quantiles:", quantiles)
# # sort the input text, output text and label text, overlap
# data_processed.sort(key=lambda x: x["overlap"], reverse=True)

# Define a function to process a single item
def postprocess_sample(i):
    output_ids = outputs[i].outputs[0].token_ids.tolist()
    label_ids = tokenized_sample_list[i]["label_ids"]
    overlap = compute_lcs(output_ids, label_ids)
    return {
        "input_text": prompts[i],
        "output_text": outputs[i].outputs[0].text,
        "label_text": tokenized_sample_list[i]["label_text"],
        "overlap": int(overlap),
        "meta": raw_data[i]["meta"],
    }

# Use multiprocessing to process items in parallel
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    data_processed = list(tqdm.tqdm(pool.map(postprocess_sample, range(len(raw_data))), total=len(raw_data), desc="Processing samples"))

# Print the average overlap
# print("Average overlap:", np.mean(overlaps))
overlaps = [item["overlap"] for item in data_processed]
quantiles = np.percentile(overlaps, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print("Quantiles:", quantiles)

# Sort the input text, output text and label text, overlap
data_processed.sort(key=lambda x: x["overlap"], reverse=True)

# save to jsonl
import json
with open(f"pile-cc-{len(data_processed) // 1000}k-{model_tag}.{split_tag}.jsonl", "w") as f:
    for sample in data_processed:
        f.write(json.dumps(sample) + "\n")

# # save the top 16k samples to jsonl
# with open("pile-llama31-8b-memorized-16k.jsonl", "w") as f:
#     for sample in data_processed[:16_000]:
#         f.write(json.dumps({"text": sample["input_text"] + sample["label_text"]}) + "\n")

# %%
# # Compute the histogram of overlaps
# plt.hist(overlaps, bins=30, edgecolor='black')
# plt.title('Histogram of Token Overlaps')
# plt.xlabel('Number of Overlapping Tokens')
# plt.ylabel('Frequency')
# plt.show()


