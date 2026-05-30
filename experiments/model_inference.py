from openai import OpenAI
from google import genai
from google.genai import types
from xai_sdk import Client as xaiClient
from xai_sdk.chat import user
from xai_sdk.tools import web_search as xai_web_search
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenAI Client Setup
client_openai = OpenAI(api_key=open("../config/openai_key.txt", encoding="utf-8").read())

# Google GenAI Client Setup
client_google = genai.Client(api_key=open("../config/google_key.txt", encoding="utf-8").read())

# XAI Client Setup
client_xai = xaiClient(api_key=open("../config/xai_key.txt", encoding="utf-8").read())

###############################################################################
def generate_with_openai(prompt, model_name="gpt-5-nano-2025-08-07", reasoning_effort=False, web_search=False):
    # default reasoning effort is medium for gpt-5 models, none for successor models like GPT-5.4 etc. Override with reasoning_effort argument.
    if reasoning_effort:
        reasoning_effort = "medium"
    # gpt-5 series minimal produce 0 thinking tokens and for successor models like gpt-5.4, for minimum reasoning effort set it to "none".
    else:
        reasoning_effort = "minimal" if "gpt-5-" in model_name else "none"
    try:
        response = client_openai.responses.create(
            model=model_name,
            tools=[{"type": "web_search"}] if web_search else [],
            reasoning={"effort": reasoning_effort},
            input=prompt
        )
    except:
        time.sleep(120)
        response = client_openai.responses.create(
            model=model_name,
            tools=[{"type": "web_search"}] if web_search else [],
            reasoning={"effort": reasoning_effort},
            input=prompt
        )

    return response

################################################################################
def generate_with_google(prompt, model_name="gemini-2.5-flash", reasoning_effort=False, web_search=False):
    tools = []
    if web_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    if reasoning_effort:
        thinking_config = types.ThinkingConfig(include_thoughts=True)
        config_google = types.GenerateContentConfig(tools=tools, thinking_config=thinking_config)
    else:
        config_google = types.GenerateContentConfig(tools=tools)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                client_google.models.generate_content,
                model=model_name,
                contents=prompt,
                config=config_google
            )
            # This waits for 60 seconds. If nothing returns, it raises an error and jumps to except
            response = future.result(timeout=60)
    except:
        time.sleep(120)
        response = client_google.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config_google
        )

    return response

################################################################################
def generate_with_xai(prompt, model_name="grok-4.3", reasoning_effort=False, web_search=False):
    reasoning_effort = "medium" if reasoning_effort else "none"
    tools = [xai_web_search()] if web_search else []

    try:
        chat = client_xai.chat.create(
            model=model_name,
            tools=tools,
            reasoning_effort=reasoning_effort
        )
        chat.append(user(prompt))
        response = chat.sample()
    except:
        time.sleep(120)
        chat = client_xai.chat.create(
            model=model_name,
            tools=tools,
            reasoning_effort=reasoning_effort
        )
        chat.append(user(prompt))
        response = chat.sample()
    
    return response

################################################################################
def generate_with_closed_model(model_name, reasoning_effort, web_search, prompts, claimIds):
    def get_resp(pr):
        if "gpt" in model_name.lower():
            return generate_with_openai(pr, model_name, reasoning_effort, web_search)
        elif "gemini" in model_name.lower():
            return generate_with_google(pr, model_name, reasoning_effort, web_search)
        elif "grok" in model_name.lower():
            return generate_with_xai(pr, model_name, reasoning_effort, web_search)
        else:
            print(f"The said model {model_name} is not supported.")
            exit()

    def extract_out(rs):
        if "gpt" in model_name.lower():
            return str(rs.output_text).strip()
        elif "gemini" in model_name.lower():
            return str(rs.text).strip()
        elif "grok" in model_name.lower():
            return str(rs.content).strip()
        else:
            print(f"The said model {model_name} is not supported.")
            exit()

    resps = [None] * len(prompts)
    # You can adjust max_workers based on your API tier and rate limits. In general, 8-16 works well for most cases.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_resp, pr): i for i, pr in enumerate(prompts)}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            resps[idx] = future.result()

    Outs, Logs = {}, {}
    for id, pr, rs in zip(claimIds, prompts, resps):
        Logs[id] = {"prompt": pr, "response": str(rs)}
        Outs[id] = extract_out(rs)

    return Outs, Logs

################################################################################
def generate_with_qwen3(model_name, reasoning_effort, prompts, claimIds):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    Outs, Logs = {}, {}
    for id, pr in tqdm(zip(claimIds, prompts), total=len(claimIds)):
        messages = [
            {"role": "user", "content": pr}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=reasoning_effort
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096, # Default as mentioned in Qwen3 docs: 32768
            do_sample=False
        )
        generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        resp = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # this will work in both cases - thinking or non-thinking!!
        Outs[id] = resp.split("</think>")[-1].strip()
        Logs[id] = {"prompt": pr, "response": resp}
        torch.cuda.empty_cache()
            
    return Outs, Logs

################################################################################
def generate_with_cot_reasoning(model_name, prompts, claimIds):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    Outs, Logs = {}, {}
    for id, pr in tqdm(zip(claimIds, prompts), total=len(claimIds)):
        # You are Phi, a language model trained by Microsoft to help users.
        messages = [
            {"role": "system", "content": "You are a language model trained to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:"},
            {"role": "user", "content": pr}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096, # Default as mentioned in Phi4 docs: 4096
            do_sample=False
        )
        generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        resp = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # this will work in both cases - thinking or non-thinking!!
        Outs[id] = resp.split("</think>")[-1].strip()
        Logs[id] = {"prompt": pr, "response": resp}
        torch.cuda.empty_cache()
            
    return Outs, Logs

################################################################################
def generate_with_if_model(model_name, prompts, claimIds):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    Outs, Logs = {}, {}
    for id, pr in tqdm(zip(claimIds, prompts), total=len(claimIds)):
        messages = [
            {"role": "user", "content": pr}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        resp = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        Outs[id] = resp
        Logs[id] = {"prompt": pr, "response": resp}
        torch.cuda.empty_cache()
            
    return Outs, Logs

################################################################################
def generate_with_open_model(model_name, reasoning_effort, prompts, claimIds):
    if "qwen3" in model_name.lower():
        return generate_with_qwen3(model_name, reasoning_effort, prompts, claimIds)
    elif reasoning_effort==True:
        return generate_with_cot_reasoning(model_name, prompts, claimIds)
    else:
        return generate_with_if_model(model_name, prompts, claimIds)