zero_shot_prompt = lambda cl, ts: f"""You are a fact-checking assistant. Your task is to classify the given claim into true or false.

Claim: {cl}
Timestamp: {ts}

Answer with the final verdict of "True" or "False" only. Do not include any explanations or additional text."""

zero_shot_web_search_prompt = lambda cl, ts: f"""You are a fact-checking assistant with access to web-search. Using evidence retrieved from web, classify the given claim as true or false.

Claim: {cl}
Timestamp: {ts}

Answer with the final verdict of "True" or "False" only. Do not include any explanations or additional text."""

zero_shot_web_search_evidence_guided_prompt = lambda cl, ts, ev: f"""You are a fact-checking assistant with access to web-search. Using evidence retrieved from web, classify the given claim as true or false.

Claim: {cl}
Timestamp: {ts}
Evidence URLs: {ev}

Prioritize the Evidence URLs when they are relevant; use web-search if they are insufficient, inaccessible, or do not address the claim.

Answer with the final verdict of "True" or "False" only. Do not include any explanations or additional text."""