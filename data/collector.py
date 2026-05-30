import pandas as pd
from tqdm import tqdm
from glob import glob
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42  # For consistent language detection results
import sys, os
from datetime import date
from utils import load_cn_data, domain_classification, date_to_millis, sample_languages
from tweet_scraper import fetch_tweet_metadata

os.makedirs("CF", exist_ok=True)

# input parameters
st_date = date_to_millis(sys.argv[1]) if len(sys.argv) > 1 else date_to_millis("2025-01-01")
end_date = date_to_millis(sys.argv[2]) if len(sys.argv) > 2 else date_to_millis("2025-12-31")
n_lang = int(sys.argv[3]) if len(sys.argv) > 3 else 5
min_lang_count = int(sys.argv[4]) if len(sys.argv) > 4 else 0
max_lang_count = int(sys.argv[5]) if len(sys.argv) > 5 else 100000

# Load the notes dataset and keeping helpful non-media notes and annotating note-language
print("Loading and filtering notes...")
df_notes = load_cn_data('notes')
df_notes = df_notes[(df_notes.createdAtMillis >= st_date) & (df_notes.createdAtMillis <= end_date)]

df_note_status = load_cn_data('noteStatusHistory')
df_note_status.index = df_note_status.noteId

tmp_note_status = []
for noteId in tqdm(df_notes.noteId):
    if noteId in df_note_status.index:
        tmp_note_status.append(df_note_status.loc[noteId].currentStatus)
    else:
        tmp_note_status.append(None)

df_notes["currentStatus"] = tmp_note_status
df_notes_helpful = (
    df_notes[df_notes['currentStatus'] == 'CURRENTLY_RATED_HELPFUL']
    .loc[lambda df: df.groupby('tweetId')['createdAtMillis'].idxmax()]
)

df_notes_helpful = df_notes_helpful[df_notes_helpful['isMediaNote'] == 0]
df_notes_helpful.reset_index(drop=True, inplace=True)

del tmp_note_status, df_note_status, df_notes

lang = []
for i in tqdm(df_notes_helpful.summary):
    try:
        lang.append(detect(i))
    except:
        lang.append(None)

df_notes_helpful["noteLanguage"] = lang

print(df_notes_helpful.noteLanguage.value_counts())

# Sampling languages with constraints
print("\nSampling languages with constraints...")
for i in df_notes_helpful.noteLanguage.value_counts().index[:n_lang]:
    cnt = df_notes_helpful.noteLanguage.value_counts()[i]
    if cnt >= max_lang_count:
        print(f"Language: {i}, Count: {cnt}, Status: Including {max_lang_count} samples")
    elif cnt >= min_lang_count:
        print(f"Language: {i}, Count: {cnt}, Status: Including all samples")
    else:
        print(f"Language: {i}, Count: {cnt}, Status: Excluding samples (very low frequecy)")

df_sampled = sample_languages(df_notes_helpful, n_lang, min_lang_count, max_lang_count)

# Domain classification using OpenAI API
print("\nClassifying domains using OpenAI API...")
df_sampled["domain"] = domain_classification(df_sampled.summary.tolist())

print(df_sampled.domain.value_counts())

df_sampled = df_sampled[
    df_sampled["domain"].str.contains(r"politics|finance", na=False)
]  # Focusing on politics and finance domains for metadata fetching

# Save the filtered and classified dataset
df_sampled.to_csv("CF/filtered_classified_notes.tsv", sep='\t', index=False)

# For fetching tweet metadata, you can use the following code snippet (make sure to have the tweet_scraper.py file with the fetch_tweet_metadata function implemented as shown in the previous code snippet):
tweetIds = df_sampled.tweetId.unique().tolist()
tweet_metadata = fetch_tweet_metadata(tweetIds, output_file="CF/tweets_metadata.json")

print(f"Featched metadata for {len(tweet_metadata)} tweets out of {len(tweetIds)} as of {date.today()}. Saved to tweets_metadata.json")