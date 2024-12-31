import pandas as pd
from query_models import query_gpt, query_claude, query_gemini

OPENAI_KEY = 'sk-None-fjK6Ygp0xLWRM33TINu3T3BlbkFJZtydkhCtFT9t9TjOH2iD'
CLAUDE_KEY = ""

generate_samples(prompts, N=100, model="gpt"):
	model_functions = {"gpt": query_gpt} #, "claude": query_claude}
	generated_df = pd.DataFrame()
	query_model = model_functions.get(model)

	for prompt, label in prompts:
		for i in range(N):
			# Isolate model to query
			generated_df.append(query_model(prompt), label=label)

if __name__ == "__main__":
    savedir = ""
    model = "gpt"
    prompts = [["Write a positive movie review", 1], ["Write a negative movie review", 0]]

    samples_df = generate_samples(prompts, N=10, model=model)
