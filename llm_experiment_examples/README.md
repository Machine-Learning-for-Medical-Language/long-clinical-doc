## LLM Experiment Examples for LCD-Benchmark Paper

This folder contains example codes for open-sourced LLMs and evaluation code for our benchmark paper.

For inference on the dataset, please use `llm_zeroshot.py`. For evaluation of the generated predictions, use `evaluate.py --pred_path <outputfolder>/predictions.txt` to see the F1 score.

#### Environments
This code was tested on Ubuntu 22.04 LTS and Rocky Linux 8.9, python 3.10, torch 2.3.1+cu121 and Transformers 4.41.2. 
For more details, please refer to `requirements.txt`.
We tested the code on single A40 and A100 GPUs by NVIDIA.

We tested the following models. Please check each model's requirements (e.g., use agreements and authentication tokens) prior to executing the code.

* [Meerkat-7b-v1.0](https://huggingface.co/dmis-lab/meerkat-7b-v1.0)
* [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
* [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
* [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
* [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
* [Qwen2-72B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-72B-Instruct-AWQ)

### Examples

Please set the data path and output path as actual paths.
```bash
export DATA_PATH=../out_hos_30days_mortality
export output_dir=./outputs-tmp/
mkdir -p $output_dir
```

The following lines will run an example experiment using Llama3-8b on the dataset and will output `predictions.txt` and `predictions.json` under `$output_dir`.
```bash
# Example LLM
export LM_NAME=meta-llama/Meta-Llama-3-8B-Instruct

export MAX_LEN=8192
export OUTPUT_BUFFER="-1"

python llm_zeroshot.py \
 --model_name_or_path ${LM_NAME} \
 --data_path ${DATA_PATH} \
 --max_length ${MAX_LEN} \
 --output_buffer ${OUTPUT_BUFFER} \
 --output_path ${output_dir}
```

Evaluate the output with following example.
```bash
# Evaluation:
python eval_predictions_binary.py --pred_path ${output_dir}/predictions.txt 
```


### Paper information
Citation information:
```
@article{yoon2024lcd,
  title={LCD Benchmark: Long Clinical Document Benchmark on Mortality Prediction for Language Models},
  author={Yoon, WonJin and Chen, Shan and Gao, Yanjun and Zhao, Zhanzhan and Dligach, Dmitriy and Bitterman, Danielle S and Afshar, Majid and Miller, Timothy},
  journal={Under review (Preprint: medRxiv)},
  pages={2024--03},
  year={2024}
}
```