# Datasets and model candidates (T0.3)

Checked 2026-11-02 from primary sources: official repos, dataset and model cards, license files, paper text, and the hosting site's own pages. Nothing was downloaded or installed. Where a primary page could not be read, the entry says "couldn't confirm".

Standing rule (`CLAUDE.md`): a dataset is used only if a primary-source license allows non-commercial research use with no signed agreement. Otherwise `Approved: no, needs the person`. The same test is applied to the model candidates below, by analogy.

Summary:
- DECODE: yes, by the person, for local, non-commercial evaluation only; never commit or redistribute its data (no data license stated).
- QMSum: yes, by the person, for local, non-commercial evaluation only; never commit or redistribute its data (data license not stated; it draws on AMI, ICSI, and two parliaments with terms not confirmed).
- LongMemEval: yes (MIT).
- AMI Meeting Corpus: yes (CC BY 4.0).
- pyannote (diarization): yes, by the person, who accepted its terms on Hugging Face (gated model).
- NVIDIA Sortformer v2 (diarization alternative): yes (CC BY 4.0).
- microsoft/deberta-v3-base (consistency base): yes (MIT).
- Qwen3-4B (small LLM): yes on license; 4-bit fit under 7 GB couldn't be confirmed.
- Qwen2.5-1.5B-Instruct (small LLM alternative): yes on license; 4-bit fit couldn't be confirmed.
- Qwen2.5-3B-Instruct: no, needs the person (custom research agreement).

## DECODE

URL: https://parl.ai/projects/contradiction/ (project page); https://github.com/facebookresearch/ParlAI/blob/main/projects/contradiction/README.md (README); paper https://arxiv.org/abs/2012.13391 (Nie et al., "I like fish, especially dolphins", ACL 2021).
License: Couldn't confirm. The parl.ai page and the README state no license or terms for the data. The paper says only that it is "publicly available at https://parl.ai/projects/contradiction". The ParlAI repo license (https://github.com/facebookresearch/ParlAI/blob/main/LICENSE) is MIT, but that is the code license, not a data license. The "Terms of Service" link on parl.ai belongs to the ParlAI website.
Size: 6 groups of dialogues (README and parl.ai): Train 27,184; Dev 4,026; Test 4,216; Human-Bot 764; A2T 2,079 and RCT 2,011 (auxiliary test sets built from Test).
Format: Couldn't confirm. The pages read do not describe the file format. Access is through ParlAI (`parlai display_data -t decode -dt train|valid|test`, with `--test_type vanilla|human-bot|a2t|rct`) or by raw S3 download (`projects/contradiction/download_with_raw_format.md` in the ParlAI repo, not read).
Labels: Confirmed binary, not three-way. Paper (render at https://ar5iv.labs.arxiv.org/html/2012.13391): "y can be 0 or 1 corresponding to the non-contradiction and the contradiction label respectively." Splits are balanced between the two labels. In the Human-Bot set, half of the 764 dialogues end with a bot utterance that contradicts itself, and the labels were checked by three extra annotators. Every A2T example is "contradiction", and every RCT example is "non-contradiction". The paper maps entailment and neutral from other NLI data to non-contradiction when that data is used for training. So the assumption "DECODE only labels contradiction vs not" is correct.
Access: ParlAI loader or the S3 raw download above. No registration or signed agreement was found in the pages read. Nothing was downloaded.
Approved: yes, by the person on 2026-11-02, for local, non-commercial evaluation only. Its data must never be committed or redistributed. Original finding: no primary source states a license or terms for the DECODE data, so the standing rule cannot be met. The person can decide whether to accept the ParlAI distribution as it is.

## QMSum

URL: https://github.com/Yale-LILY/QMSum (repo and README); paper https://arxiv.org/abs/2104.05938 (NAACL 2021).
License: Code is MIT (https://github.com/Yale-LILY/QMSum/blob/main/LICENSE, "Copyright (c) 2021 Yale-LILY"). The README and the arXiv abstract page state no license for the data. The paper's Ethics section (render at https://ar5iv.labs.arxiv.org/html/2104.05938) says: "All the meeting transcripts we collected are public and open to use according to the regulation," and it warns that "some meeting transcripts are intended for internal use only." It also links to the AMI, ICSI, Senedd and Canadian Parliament terms pages. Source corpora and their terms:
- Product meetings (137) come from AMI. AMI is CC BY 4.0 (see the AMI section).
- Academic meetings (59) come from ICSI. ICSI is CC BY 4.0 (https://groups.inf.ed.ac.uk/ami/icsi/license.shtml).
- Committee meetings: 25 from the Welsh Parliament (Senedd) and 11 from the Parliament of Canada. Couldn't confirm their transcript terms. The Senedd Commission's copyright page (https://research.senedd.wales/commission/access-to-information/copyright/) covers Senedd.tv clips, not the Record of Proceedings text, which I could not find a license for. For Canada, the committee evidence is governed by a Speaker-controlled permission, as far as I could find from older documents, and I could not confirm the current notice on the evidence PDFs.
Size: 1,808 query-summary pairs over 232 meetings in three domains (paper abstract and README).
Format: JSON and JSONL under `data/ALL`, `data/Academic`, `data/Product`, `data/Committee`, with train, val and test splits (README). Each record has `topic_list` (topic with `relevant_text_span`), `general_query_list`, `specific_query_list` (query, answer, span), and `meeting_transcripts` (speaker, content).
Labels: No classification labels. The gold items are the query-answer pairs and the topic spans in the transcript (README).
Access: Public GitHub repo, no gate. Derived from the AMI, ICSI and parliamentary sources, so the terms of those sources also apply. Nothing was downloaded.
Approved: yes, by the person on 2026-11-02, for local, non-commercial evaluation only. Its data must never be committed or redistributed. Original finding: the QMSum data itself has no stated license. Its meetings come from four sources, and two of those have terms I could not confirm. The AMI approval below does not cover QMSum's copy of the AMI transcripts.

## LongMemEval

URL: https://github.com/xiaowu0162/LongMemEval (repo and README); data https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (dataset card and API https://huggingface.co/api/datasets/xiaowu0162/longmemeval-cleaned); paper https://arxiv.org/abs/2410.10813 (ICLR 2025).
License: MIT for the repo (https://github.com/xiaowu0162/LongMemEval/blob/main/LICENSE, "Copyright (c) 2024 Di Wu"). The HF dataset card and API list `license: mit`, and the dataset is not gated. Caveat: the filler chat sessions come from ShareGPT and UltraChat. The paper's Ethics statement says ShareGPT "has an Apache 2.0 license" and UltraChat "has an MIT license", and that "In the data release, we plan to use an MIT license." These upstream terms are the authors' own statements, which I did not verify against ShareGPT's or UltraChat's sources. The MIT grant itself is confirmed on the repo and the card. The paper's sampling is "25% ShareGPT, 25% UltraChat, and 50% simulated sessions."
Size: 500 evaluation instances per file. Files: `longmemeval_oracle.json`, `longmemeval_s_cleaned.json`, `longmemeval_m_cleaned.json` (about 115k tokens of history per instance for the S file, per README). Byte sizes couldn't confirm.
Format: JSON. Fields per instance (README): `question_id`, `question_type`, `question`, `answer`, `question_date`, `haystack_session_ids`, `haystack_dates`, `haystack_sessions` (turns with role and content, and `has_answer` on evidence turns), `answer_session_ids`.
Labels: `question_type` takes six values: single-session-user, single-session-assistant, single-session-preference, temporal-reasoning, knowledge-update, multi-session. IDs ending in `_abs` are abstention questions. Turn-level evidence is in `has_answer`, and session-level evidence in `answer_session_ids` (README).
Access: Direct HTTP download from the HF URLs (README `wget` commands). No registration or signed agreement found. Nothing was downloaded.
Approved: yes. Reason: MIT on the repo and the HF card, ungated, no signed agreement. The upstream ShareGPT and UltraChat terms are noted above and are the person's call if they want a stricter reading.

## AMI Meeting Corpus

URL: https://groups.inf.ed.ac.uk/ami/corpus/ (corpus overview); license https://groups.inf.ed.ac.uk/ami/corpus/license.shtml; download https://groups.inf.ed.ac.uk/ami/download/.
License: CC BY 4.0. The license page says: "The AMI corpus and its annotations are released under the Creative Commons Attribution 4.0 license agreement," and "Use of this data implies agreement with the terms below." The download page says the manual annotations (v1.6.2) and automatic annotations (v1.5.1) were "license altered to CC BY 4.0." The corpus page says the signals and transcription, and some annotations, are under CC BY 4.0. Caveat: older text on the overview page (https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml) still mentions ShareAlike and NonCommercial CC licenses, and the corpus page has an old NC-SA 2.5 link in a comment. The current license page says CC BY 4.0, so I used that. The person may want to see this discrepancy.
Size: 100 hours of meeting recordings (corpus page). Meeting count couldn't confirm from the pages read.
Format: Audio and video signals (WAV and video), transcription, and annotations. The download page says the annotations need NXT version 1.4.4 to use.
Labels: Manual and automatic annotations are provided per meeting (download page). I did not read the annotation documentation, so the specific label types couldn't confirm.
Access: Choose meetings and file types on the download page. I could not confirm whether registration is required. A search snippet said "after registering", but no primary page I could read said so, and the license page says use implies agreement rather than a signed form. The corpus page links a blank copy of the participants' signed consent form (https://groups.inf.ed.ac.uk/ami/corpus/ethicsandconsent.shtml), so the recordings are consented, as far as that page says. A later re-fetch of the download page returned HTTP 403, so the access steps are from an earlier fetch.
Approved: yes. Reason: CC BY 4.0 allows non-commercial research, no signed agreement is required by the license page, and the participants signed consent forms, per the ethics page. This approval covers only the AMI corpus as published by AMI. It does not cover QMSum's copy.

## Diarization: pyannote (speaker-diarization-3.1)

URL: https://huggingface.co/pyannote/speaker-diarization-3.1 (model); code https://github.com/pyannote/pyannote-audio.
License: The code is MIT (https://github.com/pyannote/pyannote-audio/blob/develop/LICENSE, "Copyright (c) 2020 CNRS"). The model's HF metadata (https://huggingface.co/api/models/pyannote/speaker-diarization-3.1) says `license: mit` and `gated: auto`. The model card text couldn't be read unauthenticated (HTTP 401), so the terms that the user accepts on HF are unconfirmed. The segmentation model it uses (https://huggingface.co/api/models/pyannote/segmentation-3.0) is also MIT and gated.
Access: Gated. Accept the terms on HF and use a read token, which is the person's step in `docs/HUMAN-GATES.md`.
Approved: yes, by the person on 2026-11-02, who accepted the model's terms on Hugging Face. Original finding: the access gate requires accepting model terms I could not read, so the license cannot be confirmed from the model card. The person can accept the terms and decide.

## Diarization alternative: NVIDIA Sortformer (diar_streaming_sortformer_4spk-v2)

URL: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2 (model card and API https://huggingface.co/api/models/nvidia/diar_streaming_sortformer_4spk-v2); runtime NeMo https://github.com/NVIDIA/NeMo.
License: CC BY 4.0 on the model card (`license: cc-by-4.0`, and `license:cc-by-4.0` in the API). Not gated. NeMo code is Apache 2.0 (https://github.com/NVIDIA/NeMo/blob/main/LICENSE). The card lists AMI, ICSI and other speech sets among its training data. These are training sets for the model, not this project's evaluation data.
Access: Public HF download. Needs the NeMo runtime, which is not installed.
Approved: yes. Reason: CC BY 4.0 for the model, ungated, and the code is Apache 2.0. Not run here. Latency and GPU fit are unverified. The older sibling `nvidia/diar_sortformer_4spk-v1` is CC BY-NC 4.0 (API), which is allowed for non-commercial use but is not the choice here.

## Consistency base model: microsoft/deberta-v3-base

URL: https://huggingface.co/microsoft/deberta-v3-base (model card); API https://huggingface.co/api/models/microsoft/deberta-v3-base.
License: MIT (`license: mit` in the model card header, and in the API). Not gated.
Size: Backbone about 86M parameters plus 98M in the embedding layer (model card).
Access: Public HF download.
Approved: yes. Reason: MIT, ungated, no signed agreement.

## Small LLM: Qwen3-4B

URL: https://huggingface.co/Qwen/Qwen3-4B (model card and LICENSE link); API https://huggingface.co/api/models/Qwen/Qwen3-4B.
License: Apache-2.0 (model card header: `license: apache-2.0`, with `license_link` to the repo LICENSE; API matches). Not gated.
Size: 4,022,468,096 parameters, BF16 safetensors (API metadata).
4-bit memory: Couldn't confirm that it fits under 7 GB. I found no primary source giving a 4-bit or QLoRA memory figure. Weight arithmetic (4.0B parameters at 4 bits is about 2 GB) is my estimate, not a sourced measurement. Training overhead is unmeasured. The T-task that uses it must measure and block if it does not fit, as PLAN.md requires.
Access: Public HF download.
Approved: yes on license. Reason: Apache-2.0, ungated, no signed agreement. Fit under 7 GB is still unconfirmed (see above).

## Small LLM alternative: Qwen2.5-1.5B-Instruct

URL: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct (model card); API https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct.
License: Apache-2.0 (model card header, with `license_link` to its LICENSE; API matches). Not gated.
Size: 1,543,714,304 parameters, BF16 (API metadata).
4-bit memory: Couldn't confirm fit under 7 GB, for the same reason as Qwen3-4B.
Approved: yes on license. Reason: Apache-2.0, ungated, no signed agreement. Fit unconfirmed.

## Small LLM candidate not approved: Qwen2.5-3B-Instruct

URL: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct; license text https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE (Qwen RESEARCH LICENSE AGREEMENT, release date September 19, 2024).
License: Custom `qwen-research` license (API `license:other`). It defines "Non-Commercial" as "for research or evaluation purposes only" and says "By clicking to agree or by using or distributing any portion or element of the Qwen Materials, you will be deemed to have recognized and accepted the content of this Agreement." Commercial use requires a separate license from Alibaba Cloud.
Size: 3,085,938,688 parameters, BF16 (API metadata).
Approved: no, needs the person. Reason: it is a custom research agreement the user accepts by use, which the standing rule does not clearly allow. The person should decide. Its 4-bit fit is also unconfirmed.

## Notes
- This review downloaded and installed nothing. Later tasks download only `yes` entries.
- The `Approved:` lines above follow the standing rule. Only `yes` entries may be used by T1.1 (datasets) or T9.1 (small LLM). Qwen2.5-3B-Instruct stays unapproved.
- DECODE and QMSum rest on the person's decision, not on a stated license, so the standing rule's license test is not what approves them. Nothing containing their text may enter git or leave this machine: not scenario files, predictions or run records with quotes, fixtures copied from them, or docs. Keep their data under the git-ignored `datasets/raw/`. Committed files may hold only their ids, hashes, and counts, plus metrics computed from them.

## Local serving and fine-tuning model: Qwen3.5-4B

URL: https://huggingface.co/Qwen/Qwen3.5-4B
License: Apache-2.0 (Hugging Face API: `apache-2.0`, not gated).
Classes: none; it's a language and vision model.
Use: the local model through Ollama, as `qwen3.5:4b-q4_K_M`. See `docs/model-choice.md`.
Approved: yes. Reason: Apache-2.0, ungated, no signed agreement.

## Reranker: cross-encoder/ms-marco-MiniLM-L6-v2

URL: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 (model card); API https://huggingface.co/api/models/cross-encoder/ms-marco-MiniLM-L6-v2.
License: Apache-2.0 (model card header and API tag `license:apache-2.0`). Not gated.
Use: local cross-encoder for `Reranker` in `src/contextlens/retrieval/rerank.py`, loaded lazily. Weights were not downloaded in T2.3.
Approved: yes. Reason: Apache-2.0, ungated, no signed agreement.
