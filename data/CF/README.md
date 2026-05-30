## CommunityFact Dataset Snapshot (Base Version -- v1)

To run the experiments in this repository, you only need `cf_v1_test.csv`, which can be downloaded from our Hugging Face dataset card.

The public Hugging Face link will be added to the main `README.md` of this project once the paper is accepted.

**Folder Structure:**  
If you run the pipeline, this folder will contain the following files:
- `filtered_classified_notes` # Output after domain and language classification
- `tweets_metadata.json` # Metadata indexed by tweetId
- `cf_private.csv` # Post metadata
- `cf_v1.csv` # Full dataset
- `cf_v1_train.csv` # Training set
- `cf_v1_test.csv` # Test set

**Data Timestamp:**  
This corresponds to our released version.
- Tweets available on or before: **May 11, 2026**