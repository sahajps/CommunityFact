import pandas as pd
from model_inference import generate_with_closed_model, generate_with_open_model
from prompt_templete import *
import os
import json
import sys

os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Inputs
model_name = sys.argv[1]
prompt_type = sys.argv[2]
reasoning_effort = True if sys.argv[3]=="True" else False
web_search = True if sys.argv[4]=="True" else False
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5] if len(sys.argv) > 5 else "0"

map_model_name = {"gpt-5-nano": "gpt-5-nano-2025-08-07",
                  "gemini-2.5-flash": "gemini-2.5-flash",
                  "grok-4.3": "grok-4.3",
                  "aya-expanse-8b": "CohereLabs/aya-expanse-8b",
                  "aya-expanse-32b": "CohereLabs/aya-expanse-32b",
                  "ministral-8b": "mistralai/Ministral-8B-Instruct-2410",
                  "eurollm-9b": "utter-project/EuroLLM-9B-Instruct-2512",
                  "eurollm-22b": "utter-project/EuroLLM-22B-Instruct-2512",
                  "qwen3-14b": "Qwen/Qwen3-14B",
                  "qwen3-32b": "Qwen/Qwen3-32B"}

df = pd.read_csv("../data/CF/cf_v1_test.csv")

if prompt_type=="zero_shot":
    prompts = [zero_shot_prompt(cl, ts) for cl, ts in zip(df.claim, df.noteTimeStamp)]
elif prompt_type=="web_search":
    prompts = [zero_shot_web_search_prompt(cl, ts) for cl, ts in zip(df.claim, df.noteTimeStamp)]
elif prompt_type=="web_search_URL":
    prompts = [zero_shot_web_search_evidence_guided_prompt(cl, ts, ev) for cl, ts, ev in zip(df.claim, df.noteTimeStamp, df.evidenceURLs)]
else:
    print(f"Prompt type {prompt_type} doesn't found.")
    exit()

if any(x in model_name.lower() for x in ["gpt", "gemini", "grok"]):
    Outs, Logs = generate_with_closed_model(map_model_name[model_name], reasoning_effort, web_search, prompts, df.claimId.to_list())
else:
    Outs, Logs = generate_with_open_model(map_model_name[model_name], reasoning_effort, prompts, df.claimId.to_list())

json.dump(Outs, open(f"outputs/{prompt_type}_{model_name}_{reasoning_effort}_{web_search}.json", "w", encoding="utf-8"), indent=4)
json.dump(Logs, open(f"logs/{prompt_type}_{model_name}_{reasoning_effort}_{web_search}.json", "w", encoding="utf-8"), indent=4)