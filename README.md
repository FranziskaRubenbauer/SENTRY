Artefacts:
- Prompts for the pipeline:
	File: Harmfull_Harmless_Decorated_Prompts.csv
	Contains the 50 prompts from each category. The prompts of the column "harmful" are from the advBench dataset. They were selected by the h4rm3l-team. Original data from h4rm3l in file "benchmark-advbench-50.csv". Source: 
- The Decorator set for the attack was choosen from the file "topK-Meta-Llama-3-8B-Instruct.syn_progs.bandit_self_score.mixed.csv", which is a result of the h4rm3l analysis. 
	Original data source: ""

Here is the text in a well-structured Markdown format:

# Artifacts

## Prompts for the Pipeline

- **File:** `Harmfull_Harmless_Decorated_Prompts.csv`
- **Content:** Contains 50 prompts from each category.
    - The prompts in the **"harmful"** column are from the **advBench dataset**.
    - They were selected by the **h4rm3l team**.
    - Original data from h4rm3l in file: `benchmark-advbench-50.csv`.
- **Source:** [https://github.com/mdoumbouya/h4rm3l/tree/main/experiments/experiment_130_benchmark/data/sampled_harmful_prompts](https://github.com/mdoumbouya/h4rm3l/tree/main/experiments/experiment_130_benchmark/data/sampled_harmful_prompts)

## Decorator Set

- The decorator set for the attack was chosen from the file:  
  `topK-Meta-Llama-3-8B-Instruct.syn_progs.bandit_self_score.mixed.csv`
- This is a result of the h4rm3l analysis.
- **Original data source:** [https://github.com/mdoumbouya/h4rm3l/blob/main/experiments/experiment_130_benchmark/data/synthesized_programs_top_k/Meta-Llama-3-8B-Instruct.syn_progs.bandit_self_score.mixed.csv](https://github.com/mdoumbouya/h4rm3l/blob/main/experiments/experiment_130_benchmark/data/synthesized_programs_top_k/Meta-Llama-3-8B-Instruct.syn_progs.bandit_self_score.mixed.csv)

---

## License

The h4rm3l code was released under the **MIT License**.

---

## Pipeline Configuration

### Switching Between Similarity Metrics

To switch between **Cosine Similarity** and the **Cross-Encoder model**, use the variable:
```python
use_cosine_similarity
```

### Adjusting Maximum Output Tokens

To change the maximum number of newly generated output tokens, modify the variable:
```python
MAX_NEW_TOKENS
```

---

## Starting the Pipeline via Terminal

1. First, log in to Hugging Face:
    ```bash
    huggingface-cli login --token TODO
    ```

2. Then run the pipeline:
    ```bash
    python fullpipeline.py
    ```
