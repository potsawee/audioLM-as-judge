# Folder structure
.
├── _audios.zip
├── _raw_scores_with_metadata
│   ├── raw_scores.tsv
│   ├── raw_scores_removed_excess_gt.tsv
│   └── _split1
│       ├── raw_scores_removed_excess_gt_trainset.tsv
│       ├── raw_scores_removed_excess_gt_validset.tsv
│       └── raw_scores_removed_excess_gt_testset.tsv
├── _training_files
│   └── _split1
│       ├── _full
│       │   ├── TRAINSET
│       │   ├── VALIDSET
│       │   ├── TESTSET
│       │   ├── train_mos_list.txt
│       │   ├── valid_mos_list.txt
│       │   ├── test_mos_list.txt
│       │   ├── train_system.csv
│       │   ├── valid_system.csv
│       │   └── test_system.csv
│       └── _clean
│           ├── TRAINSET
│           ├── VALIDSET
│           ├── TESTSET
│           ├── train_mos_list.txt
│           ├── valid_mos_list.txt
│           ├── test_mos_list.txt
│           ├── train_system.csv
│           ├── valid_system.csv
│           └── test_system.csv
├── _transcript
│   ├── gather_transcripts.py
│   └── additional_sentences.txt
└── readme.txt

# Files that contain all dataset metadata
raw_scores.tsv	Contains all results from the listening test, from all locales. Each HIT's validation utterance has not been included, and the results are annotated with info on the utterance, the listener, and quality checks.
raw_scores_removed_excess_gt.tsv	The same structure as "raw_scores.tsv" but the excessive amount of scores for ground truth (natural) utterances has been removed, in order to include 17-23 scores for all dataset utterances (excessive scores were collected for the natural utterances in the listening test since each test HIT included a natural sample).
raw_scores_removed_excess_gt_{trainset,validset,testset}.tsv	"raw_scores_removed_excess_gt.tsv" was split in training-validation-test subsets of approximately 70%-15%-15%. Unseen systems, listeners and sentences are included in the test set and in the validation set.

# Files ready for training MOS prediction models (for both SOMOS-full, and SOMOS-clean subset)
{TRAIN,VALID,TEST}SET	Raw listening test results (all choices of all listeners) including fields: systemId, utteranceId, choice, listenerId
{train,valid,test}_mos_list.txt	Test scores averaged per utterance, including fields: utteranceId, mean score
{train,valid,test}_system.csv	Test scores averaged per system, including fields: systemId, mean score

# Metadata description
utteranceId	Id of the utterance, composed from sentenceId and systemId.
choice	1-5: choice of listener in a Likert-type scale from very unnatural (1) to completely natural (5).
sentenceId	Original Id of sentence text. Includes info on text domain.
systemId	000-200: 000 is natural speech, 001-200 are TTS systems.
modelId	m0-m5: m0 is natural speech, m1-m5 are TTS models.
testpageId	000-999: corresponds to HIT Id on Amazon Mechanical Turk.
locale	us (United States), gb (United Kingdom), ca (Canada): registered locale of listener on Amazon Mechanical Turk.
listenerId	Anonymized AMT worker Id.
isNative	0 (no), 1 (yes): Although only residents of the respective English locale, according to AMT’s qualification, were recruited in each test, and only native English speakers were asked to participate, a native/non-native annotation checkbox was included in each test page.
wrongValidation	0 (fails to pass quality check), 1 (passes quality check): Wrong score has been assigned to the validation sample on test page. Validation utterances and respective choices have been removed from the dataset, but the page-level validation annotation has been propagated for every choice in the page.
lowNatural	0 (fails to pass quality check), 1 (passes quality check): The score assigned to the natural sample on test page is extremely low (1 or 2).
sameScores	0 (fails to pass quality check), 1 (passes quality check): All scores on test page are identical ignoring the score of the validation utterance.
highSynthetic	0 (fails to pass quality check), 1 (passes quality check): The average score of synthetic samples on page is higher or close (down to smaller by 0.1) to the natural sample's score.
clean	0 (no), 1 (yes): Clean is the logical AND of the 4 quality checks (wrongValidation, lowNatural, sameScores, highSynthetic) that have been used in the dataset per test page in Flag 0/1 form. Thus, all test pages that have passed all 4 quality checks are considered clean. 
listenerReliability	The percentage of test pages that the listener has submitted and are considered clean, expressed in the range [0, 1].

# How to collect all dataset transcripts
Out of the total 2,000 sentences used in the SOMOS dataset, 1,681 sentences come from the Blizzard Challenge listening test sentences and 319 are from the public domain (general, wiki and LJ Speech).
In the somos/transcript folder, we provide 2 files:
additional_sentences.txt	319 sentences that are in the public domain
gather_transcripts.py	a Python script for Linux (requires wget and md5sum), which downloads the Blizzard Challenge data, collects the respective test sentences, combines them with the additional public domain sentences, and creates the SOMOS transcript files. 
*IMPORTANT: You must extract the somos/audios folder before running somos/transcript/gather_transcripts.py
**Make sure there is at least 20G space available in your disk. The py script downloads and processes the official Blizzard Challenge distributions (https://www.cstr.ed.ac.uk/projects/blizzard/data.html) which contain large audio files.
Without changing the folders' structure, run gather_transcripts.py
Depending on your Internet connection, the script may take several hours to finish.
The outputs of the script are 2 files: 
(1) all_sentences.txt: the 2,000 unique sentences used in the SOMOS dataset, and 
(2) all_transcripts.txt: the full transcript of the 20,100 SOMOS utterances, with the corresponding TTS system suffixes.
