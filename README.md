# Proving Test Set Contamination in Black Box Language Models

This repository contains code for running the Sharded Rank Comparison Test introduced in [Proving Test Set Contamination in Black Box Language Models](https://arxiv.org/abs/2310.17623), in addition to the benchmarks we used in evaluating open models, which can be used to replicate our results.

Concretely, this repository contains:

- Code for running the Sharded Rank Comparison Test, a statistical test for pre-training data contamination. 
- The exact test files for the eight datasets (ARC-Easy, BoolQ, GSM8K, LAMBADA, NaturalQA, OpenBookQA, PIQA, MMLU) used to produce the results in Table 2.
- A description and leaderboard (soon to come) of our **Contamination Detection Challenge**, with links to models trained on intentionally contaminated data, which can be used to benchmark future statistical tests for dataset contamination developed by the research community.

### Testing Models for Contamination 

To run our test, use the file `compute_sharded_comparison_test.py`:

```
NAME
    compute_sharded_comparison_test.py

SYNOPSIS
    compute_sharded_comparison_test.py MODEL_NAME_OR_PATH DATASET_PATH <flags>

POSITIONAL ARGUMENTS
    MODEL_NAME_OR_PATH
    DATASET_PATH

FLAGS
    -c, --context_len=CONTEXT_LEN
        Default: 2048
    -s, --stride=STRIDE
        Default: 1024
    -n, --num_shards=NUM_SHARDS
        Default: 50
    -p, --permutations_per_shard=PERMUTATIONS_PER_SHARD
        Default: 250
    -r, --random_seed=RANDOM_SEED
        Default: 0
    -l, --log_file_path=LOG_FILE_PATH
        Type: Optional[]
        Default: None
    -m, --max_examples=MAX_EXAMPLES
        Default: 5000
```

Here is an example which tests GPT2-XL for contamination on the BoolQ dataset:

```
python compute_sharded_stat.py gpt2-xl boolq/dev.jsonl \
--context_len 1024 \
--stride 512 \
--num_shards 50 \
--permutations_per_shard 100 \
--log_file_path "result.log"
```

- Note that `MODEL_NAME_OR_PATH` can be either the name of a model as shown on HuggingFace (e.g. `gpt2-xl`, `mistralai/Mistral-7B-v0.1`, etc) or a path to a checkpoint directory.

The test files used for generating the results in Table 2 are available in `benchmarks/`. 

### Contamination Detection Challenge

To support the development of further work on detecting pretraining data contamination, we release all of the models we trained on intentionally contaminated training data, along with the associated test sets. All of the models are trained on Wikitext data from RedPajama with multiple test files injected at various duplication rates at random positions. The benchmarks can be found in `detection_challenge_benchmarks/`. Note that the test files are derived from original test sets and then shuffled (to ensure exchangeability).

- [**Contam-1.4b**](https://huggingface.co/yonatano/contam-1.4b)

  | Name                   | Size  | Duplication Count |
  | ---------------------- | ----- | ----------------- |
  | pubmedqa.txt           | 1000  | 1x                |
  | gsm8k.jsonl            | 1319  | 1x                |
  | openbookqa.jsonl       | 2000  | 1x                |
  | piqa.jsonl             | 3084  | 1x                |
  | hellaswag.jsonl        | 10003 | 1x                |
  | truthfulqa.jsonl       | 22434 | 1x                |
  | commonsenseqa.jsonl    | 1140  | 2x                |
  | naturalquestions.jsonl | 1769  | 2x                |
  | boolq.jsonl            | 3270  | 2x                |
  | ai2arc.jsonl           | 3548  | 2x                |
  | lambada.txt            | 5153  | 2x                |
  | mnli.jsonl             | 10000 | 2x                |

- **Contam-Large** (774M Params) 

  - Injected test sets same as Contam-1.4b.

- **Contam-Medium** (355M Params)

  - Injected test sets same as Contam-1.4b.

- **Contam-Small** (124M Params)

  - Injected test sets same as Contam-1.4b.

- **Contam-1.4b-dupcount-higher**

  | Name                               | Size | Duplication Count (Higher Model) | Duplication Count (Lower Model) |
  | ---------------------------------- | ---- | -------------------------------- | ------------------------------- |
  | boolq.jsonl                        | 1000 | 1                                | 1                               |
  | hellaswag.jsonl                    | 1000 | 1                                | 1                               |
  | openbookqa.jsonl                   | 500  | 1                                | 2                               |
  | naturalquestions.jsonl             | 1000 | 10                               | 2                               |
  | mnli.jsonl                         | 1000 | 10                               | 4                               |
  | truthfulqa.jsonl                   | 1000 | 10                               | 4                               |
  | piqa.jsonl                         | 1000 | 50                               | 7                               |
  | mmlu_professional_law.jsonl        | 1533 | 50                               | 7                               |
  | mmlu_professional_psychology.jsonl | 611  | 50                               | 10                              |
  | mmlu_high_school_psychology.jsonl  | 544  | 100                              | 10                              |

  

- **Contam-1.4b-dupcount-lower**

  - Injected test sets same as **Contam-1.4b-dupcount-higher**.

