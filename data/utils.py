import pandas as pd
from openai import OpenAI
from glob import glob
import time, json
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

client_openai = OpenAI(api_key=open("../config/openai_key.txt").read().strip())

# For date conversion
def date_to_millis(date_str, fmt="%Y-%m-%d"):
    dt = datetime.strptime(date_str, fmt)
    return int(dt.timestamp() * 1000)

def millis_to_twitter_format(created_at_millis):
    dt = datetime.fromtimestamp(created_at_millis / 1000, tz=timezone.utc)
    return dt.strftime('%a %b %d %H:%M:%S +0000 %Y')

# Loading the Community Notes Dataset in a single DataFrame
def load_cn_data(type):
    files = glob(f'CN/{type}*.tsv')
    df_list = []
    for file in files:
        df = pd.read_csv(file, sep='\t', low_memory=False)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# Sampling languages with constraints
def sample_languages(df, n_lang, min_lang_count, max_lang_count, random_state=42):
    # get top n languages by frequency
    lang_counts = df['noteLanguage'].value_counts()
    top_langs = lang_counts.index[:n_lang]
    
    df = df[df['noteLanguage'].isin(top_langs)]
    
    sampled_dfs = []
    
    for lang in top_langs:
        cnt = lang_counts[lang]
        df_lang = df[df['noteLanguage'] == lang]
        
        if cnt >= max_lang_count:
            sampled = df_lang.sample(n=max_lang_count, random_state=random_state)
        elif cnt >= min_lang_count:
            sampled = df_lang
        else:
            continue  # skip low frequency
        
        sampled_dfs.append(sampled)
    
    return pd.concat(sampled_dfs, ignore_index=True)

###############################################################################
def call_openai_api(messages, model_name="gpt-5.5-2026-04-23"):
    # For gpt-5, the default reasoning effort is "medium". For others, the default reasoning effort could vary. You can adjust these defaults as needed.
    try:
        response = client_openai.responses.create(
            model=model_name,
            # reasoning={"effort": reasoning_efforts},
            input=messages
        )
    except:
        time.sleep(60)
        response = client_openai.responses.create(
            model=model_name,
            # reasoning={"effort": reasoning_efforts},
            input=messages
        )

    return response

###############################################################################
def domain_classification(texts):
    def get_domain_label(pr):
        return call_openai_api(pr).output_text.strip().lower()
    
    prompt = lambda t: f"""You classify community notes into one domain.

### TASK
Assign exactly one label: Politics, Finance, or Other.

### DEFINITIONS
Politics: Government, public policy, laws/regulation, courts, elections, political figures (in official roles), geopolitics, etc.

Finance: Markets (stocks, crypto), banking, business/earnings, personal finance (income, taxes, debt, inflation), gambling/betting, economic indicators (GDP, unemployment, interest rates, trade), etc.

Other: Anything not primarily Politics or Finance.

### EXAMPLES
"The Fed raised interest rates to fight inflation." → Finance
"Congress passed a bill capping insulin prices." → Politics
"Elon Musk's net worth dropped $10B after the tweet." → Finance
"Brazil legalized sports betting; operators now pay 12% tax." → Politics
"The WHO declared mpox a global health emergency." → Other
"NASA confirmed water ice on the Moon's south pole." → Other
"Lionel Messi won the 2022 FIFA World Cup Golden Ball." → Other

### OUTPUT
Return only one label: Politics | Finance | Other

Input: {t}
Output:"""

    prompts = [prompt(t) for t in texts]

    classifications = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(get_domain_label, pr): i for i, pr in enumerate(prompts)}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            classifications[idx] = future.result()

    return classifications

#################################################################################
def claim_extraction(texts, notes):
    def get_claims(pr):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "user", "content": pr})
        output = call_openai_api(messages).output_text.strip()
        messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": self_refine_prompt})
        output = call_openai_api(messages).output_text.strip()
        try:
            json.loads(output)  # Check if output is valid JSON
        except:
            # try a simple retry without self-refinement if JSON parsing fails, as self-refinement might sometimes over-correct and produce invalid JSON
            output = call_openai_api(messages).output_text.strip()
        messages.append({"role": "assistant", "content": output})

        return output

    prompt = lambda t, n: f"""You are building a rigorous misinformation detection dataset. Your task is to extract checkworthy factual claims from the provided Tweet and Community Note. Labels are relative to the Community Note (Currently Rated Helpful): "false" means the Tweet asserted the claim and the Note refutes it; "true" means the Note asserts the claim as fact. Do not use external knowledge or infer facts from URLs.

## Extraction & Labeling Rules
1. Tweet Claims: Extract Tweet claims only when the Note clearly addresses them. Label as "false" when the Note refutes the Tweet claim; label as "true" only when the Note clearly supports or confirms the Tweet claim. Otherwise drop.
2. Note Claims: Label as "true" only when the Note itself makes a factual correction or factual assertion that is directly checkworthy.
3. Paraphrase: Rewrite all claims in your own words. Do not use exact quotes. Claims must assert verifiable facts about the world — never about the tweet, note, or source itself. Drop any claim whose entire content is about the credibility, accuracy, sourcing, or wording of the tweet/note rather than the underlying facts.
4. Checkworthiness Scope: Keep only substantive high quality real-world facts/events. Drop metadata/status claims whose main point is who posted something, whether an account/source is official, verified, affiliated, fake, or impersonating, or whether a link/livestream/post/page is available, ended, deleted, or reposted. Do not extract these claims even when they are the main correction. Keep an attributed claim only when it states a substantive real-world action, policy, event, filing, speech, decision, date, place, person, organization, or figure independent of account/link/post status. If only metadata/status claims remain, return [].
5. Text-Only: The downstream dataset will not include attached images, videos, screenshots, audio, or other media. Drop claims that require viewing, identifying, or interpreting media, including what media shows, who appears in it, where/when it was recorded, or whether it is edited, cropped, staged, AI-generated, or mislabeled.
6. Label Precision: "false" = the Tweet asserted it, and the Note refutes it. "true" = the Note asserts it as fact. For substantive attributions, label the attribution itself: if a tweet says "X claims Y", label it "true" if the Note establishes that X made that substantive claim, even if Y is false. Do not apply this attribution rule to account/source authenticity, posting status, link status, or media-identification claims.
7. Specificity: Every claim must name the specific people, organizations, countries, jurisdictions, dates, places, figures, metrics, and issuers needed to identify the fact. Do not drop key qualifiers; for example, write "U.S. visas" rather than "visas" when the country matters. Do not add extra details unless they are necessary to disambiguate the claim.
8. Language Match: Write each claim in the language of the input text segment.

## Density & Balancing Rules
1. Information Density: Do NOT hyper-fragment the text. Merge closely related contextual details such as who, what, when, and where into a single, comprehensive claim.
2. Compactness: Keep each claim short while still being standalone and unambiguous. Do not add background, explanation, evidence, rhetorical framing, or provenance unless it is essential to the fact being checked.
3. Length Neutrality: Write true and false claims at the same level of detail. Do not make true claims longer by including extra evidence, caveats, or Note explanations.
4. Claim Balance: Extract one strongest "false" Tweet claim and one strongest "true" Note claim when both are valid.
5. No Forced Claims: If no claim satisfies all rules, return [].
6. No Duplicative Opposites: Do not output duplicate claims that merely restate the same fact with opposite labels. A true corrective claim may contradict a false Tweet claim when it adds distinct factual content.
7. Output Limit: Strictly limit your output to a few high-value, information-dense claims.

## The "Zero-Context Formula" (CRITICAL)
Every claim must be 100% standalone. A reader with zero knowledge of the tweet, note, media, URL, or other claims must fully understand it and fact-check it.
- No back-references to other claims: If two claims share a subject, repeat the full subject explicitly in both. Never use "The [noun]" to refer to something established in a prior claim.
- No generic subjects: Never use "a photo", "the video", "the account", "the claim", "the note", or similar vaguely in the claim. Inject specific names and entities directly.
- No missing qualifiers: Do not omit the country, jurisdiction, organization, issuer, date, place, figure, or metric needed to make the claim identifiable.
- No explanatory padding: Do not include why the claim matters, how it was verified, or what the Note is doing.
- Structure: [Specific Subject + Explicit Names] + [Event/Context] + [Factual Assertion]

## Output Format
Return a strict JSON list only. If no claim satisfies all rules, return [].
[
  {{
    "claim": "<paraphrased, dense, explicitly detailed, zero-context factual assertion>",
    "label": "true" or "false"
  }}
]

## Example 1
Tweet: The rumor says a major bank collapsed on a silver margin call at 2:47 AM December 28. I cannot verify that. What I can verify is more interesting. JPMorgan filed an 8K on December 27 disclosing 4.875 billion dollars in unrealized silver losses. They flipped from 200 million ounces short to 750 million ounces long physical. The largest position reversal in the history of the silver market happened in the last 30 days and nobody on financial television said a word. The rumor claims 34 billion in emergency Fed repos. Official data shows routine operations under 7 billion. Either the data is lagged or the rumor is wrong. But here is what nobody is asking. Why did JPMorgan suddenly need to own three quarters of a billion ounces of physical silver after spending 15 years on the short side. What did they see coming that made them eat a 5 billion dollar loss just to get positioned the other way. The collapse story might be fiction. The position flip is filed with the SEC. One of those facts will matter more in 90 days than the other. Stop chasing the rumor. Start asking why the smartest bank in commodities just switched sides at the worst possible price and seems fine with it. https://t.co/VZD2WlF6Ux
Community Note: No 8-K filing disclosing $4.875 billion in unrealized silver losses or a position reversal was made by JPMorgan Chase on December 27, 2025. The company's most recent 8-K, filed December 8, 2025, concerns a board member's resignation and makes no mention of silver.    https://jpmorganchaseco.gcs-web.com/sec-filings  https://jpmorganchaseco.gcs-web.com/static-files/b5460587-b02f-448d-9c50-2be5374130af
Output:
[
  {{
    "claim": "JPMorgan Chase filed a December 27, 2025 Form 8-K reporting $4.875 billion in unrealized silver losses.",
    "label": "false"
  }},
  {{
    "claim": "JPMorgan Chase's December 8, 2025 Form 8-K did not mention silver.",
    "label": "true"
  }}
]

## Example 2
Tweet: Welcome to MEXICO https://t.co/example
Community Note: The video was recorded at Pierre Elliott Trudeau Airport (YUL) in Montreal, Canada, and shows agricultural workers arriving to participate in the Mexico-Canada Seasonal Agricultural Workers Program.
Output:
[]

---
Tweet: {t}
Community Note: {n}
Output:
"""
    
    self_refine_prompt = """Audit the extracted JSON against the original Tweet, Community Note, and rules above. Make the smallest necessary changes.

Check label correctness, direct support, standalone specificity, claim completeness, scope violations, duplicates, claim diversity, and label balance. Add a claim only if a central checkworthy claim was missed. Remove claims that are unsupported, vague, over-fragmented, metadata/status-based, media-centered, or duplicative.

Prefer up to 2 distinct substantive claims, ideally one strong false Tweet claim and one strong true Note claim when both are valid. Do not force both labels.

Return [] if no claim satisfies the rules, otherwise:
[{{"claim": "...", "label": "true"}}]
"""
    
    prompts = [prompt(t, n) for t, n in zip(texts, notes)]

    rewrites = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(get_claims, pr): i for i, pr in enumerate(prompts)}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            rewrites[idx] = future.result()

    return rewrites