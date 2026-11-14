# Datasets and model candidates (T0.3)

Checked 2026-11-02 from primary sources: official repos, dataset and model cards, license files, paper text, and the hosting site's own pages. Nothing was downloaded or installed. Where a primary page could not be read, the entry says "couldn't confirm".

Standing rule: a dataset is used only if a primary-source license allows non-commercial research use with no signed agreement. Otherwise `Approved: no, needs the person`. The same test is applied to the model candidates below, by analogy.

Summary:
- DECODE: yes, by the person, for non-commercial evaluation; its text may also go to Claude through `claude -p` for grading and comparison, by the person on 2026-11-03; never commit, publish or redistribute its data (no data license stated).
- QMSum: yes, by the person, for non-commercial evaluation and for training the T6.2 adapter locally; its text may also go to Claude through `claude -p` for grading and comparison, by the person on 2026-11-03; never commit or redistribute its data, and never publish that adapter or examples derived from it (data license not stated; it draws on AMI, ICSI, and two parliaments with terms not confirmed).
- LongMemEval: yes (MIT).
- AMI Meeting Corpus: yes (CC BY 4.0). Attribution: AMI Meeting Corpus, https://groups.inf.ed.ac.uk/ami/corpus/ (text via edinburghcstr/ami on Hugging Face). AMI transcript text is committed in `datasets/ami-questions/` and in the E7 prediction files under `results/runs/`, as a known exception to the never-commit rule for QMSum (see `docs/DECISIONS.md` entry 81).
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
Format: Confirmed in T5.1 by downloading `http://parl.ai/downloads/decode/decode_v0.1.zip` (the link in `projects/contradiction/download_with_raw_format.md` in the ParlAI repo; 16.5 MB, no registration) into the git-ignored `datasets/raw/decode/`. The zip holds `decode_v0.1/{train,dev,test,human-bot,a2t,rct}.jsonl`, one JSON record per line. Fields used: `record_id` (unique per line), `conversation_id` (a string, shared by at most 2 records), `turns` (list of `{turn_id, agent_id, text, ...}`; `turn_id` equals the list index; agent_id is 0 or 1), `is_contradiction` (bool). The last turn is the statement to judge and the others are the history. Other fields (`aggregated_contradiction_indices`, `verifications`, `agents`, ...) are unused. Record counts match the README: train 27,184, dev 4,026, test 4,216, human-bot 764, a2t 2,079, rct 2,011. Observations: train, dev and test are balanced 50/50. `conversation_id` is not unique per record: each repeated id has one contradiction and one non-contradiction record with different last turns (distinct conversation_ids: train 26,252, dev 3,778, test 3,957). 22 conversation_ids appear in both dev and test, and every a2t and rct conversation_id is also in test (they are built from test). human-bot has no evidence indices. One turn in the files has empty text.
Labels: Confirmed binary, not three-way. Paper (render at https://ar5iv.labs.arxiv.org/html/2012.13391): "y can be 0 or 1 corresponding to the non-contradiction and the contradiction label respectively." Splits are balanced between the two labels. In the Human-Bot set, half of the 764 dialogues end with a bot utterance that contradicts itself, and the labels were checked by three extra annotators. Every A2T example is "contradiction", and every RCT example is "non-contradiction". The paper maps entailment and neutral from other NLI data to non-contradiction when that data is used for training. So the assumption "DECODE only labels contradiction vs not" is correct.
Access: ParlAI loader or the S3 raw download above. No registration or signed agreement was found in the pages read. Nothing was downloaded.
Approved: yes, by the person on 2026-11-02, for local, non-commercial evaluation; on 2026-11-03 the person added that its text may go to Claude through `claude -p` for grading and comparison. Its data must never be committed, published or redistributed. Original finding: no primary source states a license or terms for the DECODE data, so the standing rule cannot be met. The person can decide whether to accept the ParlAI distribution as it is.

### DECODE label map and splits (T5.1)

Source labels: `is_contradiction` true is `contradiction`, false is `non_contradiction`. The consistency checker has four labels (`consistent`, `update`, `contradiction`, `insufficient`) and DECODE is binary, so it can fill only two of them. The map used by `build_pairs`:

| DECODE label | Our label |
| --- | --- |
| `contradiction` | `contradiction` |
| `non_contradiction` | `consistent` |

Reasoning and limits:
- `contradiction` is a direct match: the last turn contradicts something earlier in the dialogue.
- `non_contradiction` is not the same as `consistent`. DECODE's non-contradictions are human-written continuations that do not conflict, so `consistent` is the nearest of our labels. But DECODE never marks a legitimate update ("we moved the deadline to Wednesday") or a case with too little evidence. Nothing in DECODE maps to `update` or `insufficient`, and none is invented. Scored on DECODE, only the binary question "contradiction or not" is meaningful.
- Training on DECODE alone (the train split) therefore teaches two labels. The only sources with `update` and `insufficient` are our written scenarios, which exist for dev and test only, so a four-label model cannot be trained on train data. "Fits and predicts all four labels" holds only for the fixture in `tests/test_consistency_baselines.py`, not for the real sources. What may train the four-way model is a decision for T5.2 and the person.
- Scenario pairs keep their own gold label and carry `source` `scenarios`; DECODE pairs carry `source` `decode`. Scenarios without a gold label (recall questions) are not pairs.

Splits: DECODE keeps its official train, dev and test files. A conversation_id found in two files (22, in dev and test) is placed wholly in test, so no conversation spans two splits, and both records of a conversation stay together. Resulting records: train 27,184, dev 4,004, test 4,238; conversation_ids: train 26,252, dev 3,756, test 3,957. The human-bot, a2t and rct sets are not used. The manifest `datasets/splits/decode.json` stores the first 16 hex characters of the sha256 of each conversation_id (so it holds none of DECODE's text) and a hash of the loaded data; `python -m contextlens.data.check_splits` checks it against the raw files.

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
Approved: yes, by the person on 2026-11-02, for local, non-commercial evaluation, and for training the T6.2 adapter locally; on 2026-11-03 the person added that its text may go to Claude through `claude -p` for grading and comparison. Its data must never be committed, published or redistributed. That adapter, QMSum text, and examples derived from it must never be published. Original finding: the QMSum data itself has no stated license. Its meetings come from four sources, and two of those have terms I could not confirm. The AMI approval below does not cover QMSum's copy of the AMI transcripts.

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
Access: Gated. Accept the terms on HF and use a read token, which is the person's step.
Approved: yes, by the person on 2026-11-02, who accepted the model's terms on Hugging Face. Original finding: the access gate requires accepting model terms I could not read, so the license cannot be confirmed from the model card. The person can accept the terms and decide.

### Chosen for the backend (T8.2)

Chosen: pyannote `speaker-diarization-3.1`, in `src/contextlens/speech/diarize.py` (`PyannoteBackend`). Reason: the person accepted its terms and put an `HF_TOKEN` in `.env`. Checked in the installed pyannote.audio 4.0.7: `Pipeline.__call__` takes `{"waveform": (channel, time) tensor, "sample_rate": int}` (`core/io.py`), and the pipeline code writes no file (the only `tempfile` uses are in training and preview code); the real-marked two-voice test (`tests/test_diarize_real.py`, file-write spies) has run (result under the 2026-11-03 run below).
- Not streaming, so it runs in the background over the retained audio (at most the last 2 minutes). A turn is sent as `unknown` (confidence 0.0); when the run finishes the stored turn gets its speaker, so cards show it. The client contract has no message that updates a sent turn, and resending the turn would show it twice in clients that append, so clients that only watch `transcript` messages keep seeing `unknown`. Showing it live needs a contract addition, which this task did not make.
- **Unblocked:** the person accepted the terms of `pyannote/speaker-diarization-community-1`, which pyannote.audio 4.0.7 loads a PLDA from even for the 3.1 pipeline. The library's own fetch ran as designed, with no patch. `PyannoteBackend()` still raises a clear RuntimeError when the terms or the token are missing, and turns then stay `unknown`.
- Older pyannote.audio does not help: 3.0.1, 3.1.1, 3.2.0, 3.3.2 and 3.4.0 were checked in a throwaway venv with `torch 2.11.0+cu128`, `transformers 5.22.1`, `numpy 2.5.4`. 3.4.0 resolves with those held but fails at import (`torchaudio.AudioMetaData` is gone from torchaudio 2.11, which torch 2.11 needs); the others use `torchaudio.set_audio_backend` or `torchaudio.backend.common`, also removed. No model was fetched in that check. Using 3.x would need an older torch and torchaudio (below 2.9), which is a decision for the person.
- Run once on 2026-11-03 on the CPU (`CUDA_VISIBLE_DEVICES` empty, because the GPU was in use), with pyannote.audio 4.0.7: `tests/test_diarize_real.py` passed. It's fast enough on CPU to use in the background. Measured timings belong in a run record; E7 reports them. Earlier CPU timings came from a patched path and were dropped.
- Install: `pip install -c constraints pyannote.audio` with `torch==2.11.0+cu128`, `transformers==5.22.1`, `numpy==2.5.4` held (without the constraint pip would move torch to 2.15). It adds torchaudio 2.11.1, lightning and others; `pip check` is clean. Extra: `pip install -e .[diarize]`.
- The model loads once per conversation in the background; if it fails, the type of the error is logged and turns stay `unknown`. Clients are not told, since the contract has no message for it.
- The token is read from `HF_TOKEN` in the environment or `.env`, only when the backend loads. Weights are in the Hugging Face cache, not in git.

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
- DECODE and QMSum rest on the person's decision, not on a stated license, so the standing rule's license test is not what approves them. Nothing containing their text may enter git or be published or redistributed: not scenario files, predictions or run records with quotes, fixtures copied from them, or docs. Keep their data under the git-ignored `datasets/raw/`. The person decided on 2026-11-03 that their text may go to Claude through `claude -p` for grading and comparison (it was local-only before, so records made before that date never sent this text); that adds no right to publish, and "never publish" for the QMSum adapter and examples derived from QMSum is unchanged. Publisher-provided labels count as reviewed for claims, and a claim from them must say "on published labels". Committed files may hold only their ids, hashes, and counts, plus metrics computed from them.

## Local serving and fine-tuning model: Qwen3.5-4B

URL: https://huggingface.co/Qwen/Qwen3.5-4B
License: Apache-2.0 (Hugging Face API: `apache-2.0`, not gated).
Classes: none; it's a language and vision model.
Use: the local model through Ollama, as `qwen3.5:4b-q4_K_M`. See `docs/model-choice.md`.
Approved: yes. Reason: Apache-2.0, ungated, no signed agreement.

## Zero-shot NLI model: cross-encoder/nli-deberta-v3-xsmall (T5.1)

URL: https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall (model card); API https://huggingface.co/api/models/cross-encoder/nli-deberta-v3-xsmall.
License: Apache-2.0 (card header and API tag `license:apache-2.0`). Not gated. About 70.8M parameters (API). Base model microsoft/deberta-v3-xsmall. Checked 2026-11-02.
Training data (card): SNLI and MultiNLI. Their own licenses were not checked here; only the model weights are used.
Use: `ZeroShotChecker` in `src/contextlens/consistency/baselines.py`, loaded lazily through sentence-transformers. It scores contradiction, entailment and neutral, so the checker can output only `contradiction` or `consistent`. Not downloaded or run in T5.1.
Approved: yes. Reason: Apache-2.0, ungated, no signed agreement.

## Reranker: cross-encoder/ms-marco-MiniLM-L6-v2

URL: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 (model card); API https://huggingface.co/api/models/cross-encoder/ms-marco-MiniLM-L6-v2.
License: Apache-2.0 (model card header and API tag `license:apache-2.0`). Not gated.
Use: local cross-encoder for `Reranker` in `src/contextlens/retrieval/rerank.py`, loaded lazily. Weights were not downloaded in T2.3.
Approved: yes. Reason: Apache-2.0, ungated, no signed agreement.

## Embedding model: nomic-embed-text

URL: https://ollama.com/library/nomic-embed-text (Ollama tag list; tags `latest`, `v1.5`, `137m-v1.5-fp16`, each 274MB); https://ollama.com/library/nomic-embed-text:v1.5 (license blob `c71d239df917`); primary model card https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 (the Ollama base page links to the `nomic-ai` Hugging Face org).
License: Apache License 2.0. The Ollama v1.5 page shows "Apache License Version 2.0, January 2004" as the license text, but only the opening lines, so the full text was not read there. The Hugging Face card header says `license: apache-2.0`. No gate or signed agreement appeared on the card. Couldn't confirm: that the Ollama `latest` tag is identical to `v1.5` (only the listed sizes match), and the full Ollama license text.
Use: the default embedder (`Embedder`, `src/contextlens/llm/client.py`, default `model="nomic-embed-text"`), served locally by Ollama, as named in `docs/PLAN.md`.
Approved: yes. Reason: Apache 2.0 is a standard license that allows research use, with no signed agreement and no custom terms accepted by use.

## Judge model: gemma4:e4b-it-qat

URL: https://ollama.com/library/gemma4:e4b-it-qat (Ollama tag page, license blob `0d542e0c8804`); primary license text https://ai.google.dev/gemma/docs/gemma_4_license (linked as "Apache 2.0" from the model card); model card https://huggingface.co/google/gemma-4-E4B-it (header `license: apache-2.0`); Google's Gemma Terms of Use https://ai.google.dev/gemma/terms.
License: Apache License 2.0 for Gemma 4. The Ollama tag page shows "Apache License Version 2.0, January 2004" for this tag, and the Google Gemma 4 license page at the URL above reads as the Apache 2.0 text. The Hugging Face card says `License: apache-2.0`. The older Gemma Terms of Use page (custom, accepted by use) says: "The terms below apply to Gemma models listed in the Appendix" and "For Gemma 4 terms, see the Gemma 4 license." So the custom Gemma terms do not govern Gemma 4 per Google. No signed agreement is required on any of these pages; the Gemma 4 license page mentions none. Couldn't confirm: that the Hugging Face model card covers the exact QAT build that Ollama serves (I did not find a separate QAT card); a secondary site (canirun.ai) lists Gemma 4 under the custom Gemma terms, which conflicts with the primary pages above and was not used.
Use: the separate local judge where the writer and judge must differ (`docs/TESTING.md`, `docs/model-choice.md`). Served locally by Ollama.
Approved: yes. Reason: Apache 2.0 for Gemma 4 per the Google license page, the Ollama license blob, and the Hugging Face card. It allows research use, requires no signed agreement, and is not a custom agreement accepted by use. The QAT-build confirmation above is the remaining gap.

## Speech model: whisper.cpp ggml-tiny.en (T8.1)

URL: https://huggingface.co/ggerganov/whisper.cpp (model files, pinned at revision 5359861c739e955e79d9a303bcbc70fb988958b1, file `ggml-tiny.en.bin`, sha256 921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f); runtime https://github.com/ggml-org/whisper.cpp (tag v1.7.6); original model https://github.com/openai/whisper.
License: MIT. The Hugging Face API for the model repo lists `license: mit` and the repo is not gated; the whisper.cpp LICENSE is MIT ("Copyright (c) 2023-2026 The ggml authors") and the OpenAI Whisper LICENSE is MIT ("Copyright (c) 2022 OpenAI"). Checked 2026-11-02.
Access: ungated HTTP download, no registration or signed agreement. The Dockerfile downloads it at build time and checks the sha256; the weights are not in git.
Approved: yes. Reason: MIT, ungated, no signed agreement.

## AMI audio for E7: Hugging Face edinburghcstr/ami

URL: https://huggingface.co/datasets/edinburghcstr/ami (API https://huggingface.co/api/datasets/edinburghcstr/ami).
License: CC BY 4.0 on the dataset card (`license: cc-by-4.0`), not gated; author edinburghcstr. Pinned to revision 46f28f2503e2ec48f8867a84eef356c70476beab.
Labels: none used from it; only the audio and the begin time and speaker of each utterance. Questions and transcript still come from the AMI annotations.
Access: public HTTP range requests; no token. Configs `ihm` (15.2 GB, one headset microphone per speaker) and `sdm` (13.8 GB, one far microphone), parquet shards with an `audio` WAV per utterance.
Approved: yes, by the person on 2026-11-03, as the replacement audio source for E7 because the AMI mirror and browser access were refused (intermittent 403).
Use: `tools/hf_audio.py` reads only the row groups that hold the 11 E7 meetings and writes `datasets/raw/ami-audio/<meeting>.wav` (git-ignored) and a manifest. The audio is rebuilt from segmented utterances: gaps between utterances are silence, so latency and diarization error may be easier than on the original recordings. Some utterances are 32-bit float or stereo with identical channels; both are converted as `tools/hf_audio.py` documents. Nothing is committed.

## Speech models for E7's WER control: openai/whisper-tiny.en and whisper-base.en (Hugging Face)

URL: https://huggingface.co/openai/whisper-tiny.en (revision 87c7102498dcde7456f24cfd30239ca606ed9063, `model.safetensors` 151 MB) and https://huggingface.co/openai/whisper-base.en (revision 911407f4214e0e1d82085af863093ec0b66f9cd6, `model.safetensors` 290 MB).
License: Apache-2.0 on both model cards (Hugging Face API), not gated. Checked 2026-11-06.
Use: a diagnostic only (`experiments/e7_wer_control.py`): the weights are read as safetensors with no remote code, cached under `models/whisper-control/` (git-ignored), never committed. They are the transformers form of the same OpenAI models as the service's ggml `tiny.en`, not the service's file.
Approved: yes, by the coordinator, 2026-11-06 (base.en for comparison, if small).
