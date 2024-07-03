# Authors: LCD Benchmark paper authors (Yoon et al. 2024)
#
# Please cite https://pubmed.ncbi.nlm.nih.gov/38585973/ or
# https://www.medrxiv.org/content/10.1101/2024.03.26.24304920
#
# This code is open-source, but this does not mean that the connected resources are open-source. 
# For example, you must obtain permission for MIMIC datasets separately from https://physionet.org/


import json, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
import argparse
import logging
from datetime import datetime
import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print(f"Log will show from Level: {logger.getEffectiveLevel()}")


def read_data_path(data_path:str) -> dict:
    if os.path.isdir(data_path):
        data_path = os.path.join(data_path, 'test.json')
    elif not(os.path.exists(data_path)):
        raise OSError(f"{data_path} not exist!")

    if data_path.lower().endswith(".json"):
        pass
    else:
        raise NotImplementedError(f"File should be in json format. {data_path}")
    
    # Load and parse the JSON file
    logger.info(f"Loading {data_path} ...")
    with open(data_path) as f:
        data = json.load(f)

    return data


def custom_make_sequence(input_text):
    raise NotImplementedError


def extract_first_binary(s: str) -> str:
    """Extract the first occurrence of 0 or 1 in the given string."""
    for char in str(s):
        if char in ['0', '1']:
            return int(char)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM zero-shot inference."
    )   

    parser.add_argument(
        "--model_name_or_path", required=True, help="Hugging Face Hub or local path"
    )   
    parser.add_argument(
        "--data_path", required=True, help="Path to a folder or json file"
    )   
    parser.add_argument(
        "--max_length", 
        required=False,
        type=int,
        default=8192, 
        help="Maximum length of tokens in input to process. Default:8192. Input instances longer than this limit will be truncated, meaning the tokens at the end will be dropped."
    )   
    parser.add_argument(
        "--output_buffer", 
        required=False,
        type=int,
        default=7, 
        help="Maximum length of tokens for output placeholder. If the input is longer than max_length, output_buffer will be hold-out for answer generation. (Default: 7) If set as -1, it will set to fit models behaviour."
    )   
    parser.add_argument(
        "--output_path",
        required=False,
        help="Path to save outputs. Default: os.path.join('output-' + args.model_name_or_path.replace('/', '-'))",
    )   

    args = parser.parse_args()
    logger.warning(f"### args: {args}")


    if args.output_path:
        OUTPUT_DIR = args.output_path
    else:
        OUTPUT_DIR = os.path.join("output-" + args.model_name_or_path.replace("/", "-"))
        logger.warning(f"OUTPUT_DIR set to {OUTPUT_DIR}")

    if os.path.exists(OUTPUT_DIR) and os.path.isdir(OUTPUT_DIR):
        logger.warning(f"Path {OUTPUT_DIR} seems to exist! Not making new dir")
    else:
        os.makedirs(OUTPUT_DIR)
    
    data = read_data_path(args.data_path)

    # You can override apply_chat_template with custom_make_sequence
    # 1. Define custom_make_sequence function 
    # 2. make_sequence = custom_make_sequence
    make_sequence = None # Defalut: None 

    # Model-specific Flags
    USE_FAST = True
    SYSTEM_EXIST = True
    
    # Load the model and prepare generate args
    if "llama-3" in args.model_name_or_path.lower():
        pass
    elif "mistralai" in args.model_name_or_path:
        SYSTEM_EXIST = False
    elif "meerkat-7b-v1.0" in args.model_name_or_path:
        USE_FAST = False
    elif "Qwen2".lower() in args.model_name_or_path.lower():
        if "instruct" not in args.model_name_or_path.lower():
            raise NotImplementedError(f"Instruction version of the model need to be used.")
    else:
        raise NotImplementedError(f"Model {args.model_name_or_path} not suppoted. Change the code to support the model")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        device_map="auto", 
        torch_dtype="auto", 
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=USE_FAST)

    tokenizer_kwargs = {'max_length':args.max_length, 'truncation':True, 'return_tensors':'pt'}
    
    # OUTPUT_BUFFER give space for Answer and EOS 
    # Default is set to be 7 
    # (e.g. ' 0:alive </s>' takes 7)
    if args.output_buffer == -1:
        OUTPUT_BUFFER = 7
    else:
        OUTPUT_BUFFER = args.output_buffer

    if make_sequence == None:
        def hf_apply_chat_templet(input_text):
            system_prompt = "Below is a clinical document, please remember the following clinical context and answer how likely is the given patient's out hospital mortality in 30 days?"
            question = "How likely is the given patient\'s out hospital mortality in 30 days? Please only use to answer with one word: 0:alive, 1:death"
            prompt = 'Here is the clinical document: \n ' + input_text +'\n' + question

            if SYSTEM_EXIST:
                chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                ]
            else:
                chat = [{"role": "user", "content":system_prompt + "\n" +prompt}]

            # TODO: use max_length and tokenize
            return tokenizer.apply_chat_template(chat, tokenize=False)
        make_sequence = hf_apply_chat_templet

    logger.warning(f"## model.dtype: {model.dtype}")
    logger.warning(f"## OUTPUT_BUFFER: {OUTPUT_BUFFER}")
    logger.warning(f"## Prompt: {make_sequence('INPUT_TEXT')}")
    logger.info(f"## tokenizer_kwargs: {tokenizer_kwargs}")

    with open(os.path.join(OUTPUT_DIR, "experiment_settings.txt"), 'w') as fp:
        fp.write(f"## model.dtype: {model.dtype}\n")
        fp.write(f"## tokenizer_kwargs: {tokenizer_kwargs}\n")
        fp.write(f"## OUTPUT_BUFFER: {OUTPUT_BUFFER}\n")
        fp.write(f"## Prompt example:\n {make_sequence('INPUT_TEXT')}\n")
        fp.write(f"## args: {args}\n")
        fp.write(f"## OUTPUT_DIR: {OUTPUT_DIR}\n")

    generate_text = pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,
        task='text-generation',
        device_map='auto'
    )

    torch.no_grad() # This code is for inference only
    logger.warning("## No grad enabled! As this code is for inference only")

    exe_time = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.warning(f"## Start inference on {exe_time}")

    output_dict = {
        "metadata":{"args":str(args), "date":exe_time}, 
        "data":[]
    }

    prompt_length = len(tokenizer.encode(make_sequence('')))

    # Open files to write predictions txt and json
    with open(os.path.join(OUTPUT_DIR, 'predictions.txt'), 'w') as file:
        for index, instance in tqdm.tqdm(enumerate(data['data']), total=len(data['data'])):
            input_sequence = make_sequence(instance['text'])
            tokenized_input = tokenizer.encode(input_sequence)
            len_input = len(tokenized_input)
            if len_input + OUTPUT_BUFFER >= args.max_length:
                trimed_input_code = tokenizer.encode(instance['text'])[1:1+args.max_length-prompt_length-OUTPUT_BUFFER] 
                input_sequence = make_sequence(tokenizer.decode(trimed_input_code))
                tokenized_input = tokenizer.encode(input_sequence)
            prediction = generate_text(input_sequence, **tokenizer_kwargs)[0]['generated_token_ids']
            
            output = {
                'id': instance['id'], 
                'label': instance['out_hospital_mortality_30'], 
                'prediction': tokenizer.decode(prediction),
                'binary': extract_first_binary(tokenizer.decode(prediction[min(len_input-1, args.max_length-OUTPUT_BUFFER-1):])),
                'len_input': len_input,
            }
            output_dict['data'].append(output)
            
            file.write(str(json.dumps(output)) + '\n')
            
            if index % 20 == 0:
                file.flush()
        file.flush()

    json.dump(obj=output_dict, fp=open(os.path.join(OUTPUT_DIR, 'predictions.json'), 'w'), indent =2)

    logger.info(f"Completed! {str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
    logger.info(f"Use evaluate.py --pred_path <outputfolder>/predictions.txt to see f1 score.")
