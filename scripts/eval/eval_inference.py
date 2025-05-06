import json
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM
from vllm import SamplingParams
import json
from utils import get_character_level_lcs, get_word_level_lcs, get_token_level_lcs

def load_data(args):
    if args.dataset == "extract":
        DATA_ROOT = "../../../data"
        prompt_data_path = f"{DATA_ROOT}/train-data-extract/process/{args.split}/concatenated_prefix_{args.split}.json"
        label_data_path = f"{DATA_ROOT}/train-data-extract/process/{args.split}/concatenated_suffix_{args.split}.json"

        # Load data
        with open(prompt_data_path, 'r') as file:
            prompt_data = json.load(file)

        with open(label_data_path, 'r') as file:
            label_data = json.load(file)
        
        data = [
            {
                "input": x["text"], "reference": y["text"]
            }
            for x, y in zip(prompt_data, label_data)
        ]
    # elif args.dataset == "bookmia":
    #     DATA_ROOT = "../../../data"
    #     data_path = f"{DATA_ROOT}/copyright-data/data_literal.json"
    #     with open(data_path, 'r') as file:
    #         data = json.load(file)
    
    # elif args.dataset == "newsqa":
    #     import datasets
    #     dataset = datasets.load_dataset("boyiwei/CoTaEval", "news_infringement", split="blocklisted")
    #     data = [
    #         {
    #             "input": x["prompt_autocomplete"], "reference": x["gt_autocomplete"]
    #         }
    #         for x in dataset
    #     ]
    elif args.dataset == "booksum":
        import datasets
        dataset = datasets.load_dataset("boyiwei/CoTaEval", "books_infringement", split="blocklisted")
        data = [
            {
                "input": x["prompt_autocomplete"], "reference": x["gt_autocomplete"]
            }
            for x in dataset
        ]
    # elif args.dataset == "nonadver":
    #     # non-adver.json
    #     with open("non-adver.json", 'r') as file:
    #         data = json.load(file)
    #     # empty reference
    #     data = [
    #         {
    #             "input": x["input"], "reference": "",
    #             "metainfo": {"text_type": x["text_type"], "type": x["type"]}
    #         }
    #         for x in data
    #     ]
    elif args.dataset == 'ci':
        with open("creativity_index_prompts.json", 'r') as file:
            data = json.load(file)
        # empty reference
        data = [
            {
                "input": x["input"], "reference": "",
                "metainfo": {"type": x["type"]}
            }
            for x in data
        ]
    elif args.dataset == "shield":
        import os, glob
        data_path_poem = "../../../libs/SHIELD/shield/datasets/bep/*.txt"
        data_path_book = "../../../libs/SHIELD/shield/datasets/bsnc/*.txt"
        data = []
        for file_path in glob.glob(data_path_poem):
            with open(file_path, 'r') as file:
                text = file.read()
            title = os.path.basename(file_path).removesuffix(".txt")
            data.append({
                "input": f"Provide the poem {title}:",
                "reference": text,
                "metainfo": {"type": "poem_direct", "title": title},
            })
            # if len(text.split()) > 100:
            #     data.append({
            #         "input": f"Complete the text: " + " ".join(text.split()[:50]),
            #         "reference": " ".join(text.split()[50:]),
            #         "metainfo": {"type": "poem_prefix", "title": title},
            #     })
        for file_path in glob.glob(data_path_book):
            with open(file_path, 'r') as file:
                text = file.read()
            title = os.path.basename(file_path).removesuffix(".txt")
            data.append({
                "input": f"Provide the first 100 words of the book {title}:",
                "reference": " ".join(text.split()[:100]),
                "metainfo": {"type": "book_direct", "title": title},
            })
            # if len(text.split()) > 100:
            #     data.append({
            #         "input": f"Complete the text: " + " ".join(text.split()[:50]),
            #         "reference": " ".join(text.split()[50:]),
            #         "metainfo": {"type": "book_prefix", "title": title},
            #     })
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")

    return data

def load_prompt(data, args):
    prompt_text = []

    if args.enable_chat:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = None

    for item in data:
        input_text = item["input"]

        if args.prompt_inst == "completion":
            from utils_prompt import prompt_instruction_completion
            input_text = prompt_instruction_completion() + "\n" + input_text
        elif args.prompt_inst == "none":
            pass
        else:
            raise ValueError(f"Prompt instruction {args.prompt_inst} not recognized.")

        if args.prompt_sys == "copyyes":
            assert args.enable_chat
            messages = [
                {"role": "system", "content": "Copying: Yes"},
                {"role": "user", "content": input_text},
            ]
        elif args.prompt_sys == "copyno":
            assert args.enable_chat
            messages = [
                {"role": "system", "content": "Copying: No"},
                {"role": "user", "content": input_text},
            ]
        else:
            # FIXME: the system prompt should be included in the system special token
            if args.prompt_sys == "specific":
                from utils_prompt import prompt_system_specific
                # input_text = prompt_system_specific() + "\n" + input_text
                messages = [
                    {"role": "system", "content": prompt_system_specific()},
                    {"role": "user", "content": input_text},
                ]
            elif args.prompt_sys == "assistant":
                from utils_prompt import prompt_system_assistant
                # input_text = prompt_system_assistant(args.model_tag) + "\n" + input_text
                messages = [
                    {"role": "system", "content": prompt_system_assistant(args.model_tag)},
                    {"role": "user", "content": input_text},
                ]
            elif args.prompt_sys == "none":
                # pass
                messages = [
                    {"role": "user", "content": input_text},
                ]
            else:
                raise ValueError(f"Prompt system {args.prompt_sys} not recognized.")

        if args.enable_chat:
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            input_text = "\n".join([x["content"] for x in messages])

        prompt_text.append(input_text)

    return prompt_text


def main(args):
    # %%
    if args.model_tag is None:
        args.model_tag = {
            "allenai/llama-3-tulu-2-8b": "llama3-8b-tulu",
            "meta-llama/Meta-Llama-3-8B": "llama3-8b",
        }[args.model_name]
    
    if args.model_name == "swj0419/llama2-7b_chat_newsqa" and args.tokenizer_name is None:
        args.tokenizer_name = "meta-llama/Llama-2-7b-chat-hf"
    elif args.tokenizer_name is None:
        args.tokenizer_name = args.model_name
    
    # FIXME: hard code max tokens for now
    args.max_tokens, args.min_tokens = {
        "extract": (64, 64),
        "bookmia": (64, 64),
        "newsqa": (512, 512),
        "booksum": (512, 512),
        "nonadver": (1024, 0),
        "ci": (288, 0),
        "shield": (256, 0),
    }[args.dataset]

    data = load_data(args)
    # prompt_data = prompt_data[:100]
    # label_data = label_data[:100]

    prompt_text = load_prompt(data, args)

    
    # Initialize the model
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    llm = LLM(model=args.model_name, tokenizer=args.tokenizer_name, max_model_len=16384)

    # try:
    #     llm = LLM(model=args.model_name, tokenizer=args.tokenizer_name)
    # except Exception as e:
    #     raise e


    # if args.enable_chat:
    #     prompt_text = [
    #         tokenizer.apply_chat_template([{"role": "user", "content": x["input"]}], add_generation_prompt=True, tokenize=False)
    #         for x in data
    #     ]
    # else:
    #     prompt_text = [x["input"] for x in data]
    # prompt_text

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=args.max_tokens, min_tokens=args.min_tokens, temperature=args.temperature, n=1, top_p=args.top_p)
    outputs = llm.generate(prompts=prompt_text, sampling_params=sampling_params)
    generated_text = [output.outputs[0].text for output in outputs]

    # # Example usage
    # generated_text = "The quick brown fox jumps over the lazy dog."
    # label_text = "The quick brown fox leaps over the lazy dog."
    # k = 10
    # lcs_length = compare_texts(generated_text, label_text, k, tokenizer)
    # print(f"Longest common subsequence length: {lcs_length}")

    data = [
        {
            "prompt_text": prompt,
            "label_text": item["reference"],
            "generated_text": generated,
            "lcs_char": get_character_level_lcs(generated, item["reference"]),
            "lcs_token": get_token_level_lcs(generated, item["reference"]),
            "lcs_word": get_word_level_lcs(generated, item["reference"]),
            "metainfo": item.get("metainfo", {})
        }
        for prompt, item, generated in zip(prompt_text, data, generated_text)
    ]

    # %%
    # Compute the LCS for each pair of generated and label texts
    lcs_char_all = [item["lcs_char"] for item in data]
    lcs_token_all = [item["lcs_token"] for item in data]
    lcs_word_all = [item["lcs_word"] for item in data]

    # Print the LCS lengths
    # print(lcs_lengths)
    # print 50, 75, 90, 95, 99 percentiles
    # print(np.percentile(lcs_lengths, [50, 75, 90, 95, 99]))

    # %%

    save_object = {
        **{f"lcs_char_p{p}": np.percentile(lcs_char_all, p) for p in [50, 75, 90, 95, 99]},
        **{f"lcs_token_p{p}": np.percentile(lcs_token_all, p) for p in [50, 75, 90, 95, 99]},
        **{f"lcs_word_p{p}": np.percentile(lcs_word_all, p) for p in [50, 75, 90, 95, 99]},

        "data": data,
    }

    # Save the data to a JSON file
    OUTPUT_ROOT = "./outputs"
    output_file_path = f"{OUTPUT_ROOT}/output.{args.dataset}-{args.split}.{args.model_tag}.chat-{args.enable_chat}-sys-{args.prompt_sys}-inst-{args.prompt_inst}.temp-{args.temperature:.1f}-topp-{args.top_p:.1f}.json"
    with open(output_file_path, 'w') as file:
        json.dump(save_object, file, indent=4)

    print(f"Data saved to {output_file_path}")

if __name__ == "__main__":
    # # %%
    # # model_name = "allenai/llama-3-tulu-2-8b"
    # model_name = "meta-llama/Meta-Llama-3-8B"

    import argparse
    parser = argparse.ArgumentParser()
    ## Data
    parser.add_argument("--dataset", type=str, choices=["extract", "bookmia", "newsqa", "booksum", "nonadver", "ci", "shield"], required=True)
    parser.add_argument("--split", type=str, default="dev")
    ## Model
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--model_tag", type=str)
    # Prompts
    parser.add_argument("--enable_chat", action="store_true")
    parser.add_argument("--prompt_sys", type=str, default="none")
    parser.add_argument("--prompt_inst", type=str, default="none")
    # Generation
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--min_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    main(args)
