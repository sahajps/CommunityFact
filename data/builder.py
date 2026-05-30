import pandas as pd
import json
from utils import millis_to_twitter_format, claim_extraction
from urlextract import URLExtract

url_extractor = URLExtract()

df = pd.read_csv("CF/filtered_classified_notes.tsv", sep='\t')

tweets = json.load(open("tweets_metadata.json", "r"))

df = df[df.tweetId.astype(str).isin(tweets.keys())]  # Keep only rows with available tweet metadata

tweetText, tweetTimeStamp, tweetLanguage = [], [], []
for idx, row in df.iterrows():
    tid = str(row['tweetId'])
    tweetText.append(tweets[tid]['text'])
    tweetTimeStamp.append(tweets[tid]['createdAt'])
    tweetLanguage.append(tweets[tid]['lang'])

df['tweetText'] = tweetText
df['tweetTimeStamp'] = tweetTimeStamp
df['tweetLanguage'] = tweetLanguage

df = df[df.noteLanguage == df.tweetLanguage]

print("\nExtracting claim-labels from tweet-note pairs...\n")
claims_labels = claim_extraction(df['tweetText'].tolist(), df['summary'].tolist())

df["claims_labels"] = claims_labels

cf_data = []
for idx, row in df.iterrows():
    try:
        for i, clm in enumerate(json.loads(row["claims_labels"]), start=1):
            tmp = {
                "claimId": f"cid_{row['tweetId']}_{row['noteId']}_{i}",
                "tweetId": f"tid_{row['tweetId']}",
                "noteId": f"nid_{row['noteId']}",
                "tweetText": row["tweetText"],
                "noteText": row["summary"],
                "noteTimeStamp": millis_to_twitter_format(row["createdAtMillis"]),
                "claim": clm["claim"],
                "language": row["tweetLanguage"],
                "domain": row["domain"],
                "label": clm["label"],
                "evidenceURLs": url_extractor.find_urls(row["summary"])
            }

            cf_data.append(tmp)
    except:
        # In case of any error (e.g., JSON parsing), skip this row
        pass

cf_df = pd.DataFrame(cf_data)

# The below can be used for human eval, it consists of tweet and note text
cf_df.to_csv("CF/cf_v1_private.csv", index=False)
# cf_df.to_excel("CF/cf_v1_private.xlsx", index=False)

cf_df = cf_df.drop(columns=['tweetText', 'noteText'])

# Saving the public version
cf_df.to_csv("CF/cf_v1.csv", index=False)
# cf_df.to_excel("CF/cf_v1.xlsx", index=False)