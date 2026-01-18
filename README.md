Artefacts:
- Prompts for the pipeline:
	File: Harmfull_Harmless_Decorated_Prompts.csv
	Contains the 50 prompts from each category. The prompts of the column "harmful" are from the advBench dataset. They were selected by the h4rm3l-team. Original data from h4rm3l in file "benchmark-advbench-50.csv". Source: https://github.com/mdoumbouya/h4rm3l/tree/main/experiments/experiment_130_benchmark/data/sampled_harmful_prompts
- The Decorator set for the attack was choosen from the file "topK-Meta-Llama-3-8B-Instruct.syn_progs.bandit_self_score.mixed.csv", which is a result of the h4rm3l analysis. 
	Original data source: "https://github.com/mdoumbouya/h4rm3l/blob/main/experiments/experiment_130_benchmark/data/synthesized_programs_top_k/Meta-Llama-3-8B-Instruct.syn_progs.bandit_self_score.mixed.csv"

The h4rm3l Code was released under the MIT License.

To switch between Cosine Similarity and Cross Encoder model, use the variable "use_cosine_similarity".

To change the maximum amount of new generated output tokens, change the variable "MAX_NEW_TOKENS".