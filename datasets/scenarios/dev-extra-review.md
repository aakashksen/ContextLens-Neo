# Extra dev cases review sample (15 pairs, seed 13)

Each pair is an insufficient-evidence case and its answerable twin. Check that the first really has no
answer up to as_of_seq (by reading or by inference), that the twin's answer is stated, and that the
gold fields are right. Mark any that are not.

## 1. syn-xtra-topic-no-detail-02-ins
- shape: topic-no-detail (the topic is mentioned, the asked detail never given); category: insufficient_evidence
- t0 [Imani]: The new chairs in the corner are surprisingly comfortable.
- t1 [Hollis]: Somebody put a bowl of oranges in the kitchen.
- t2 [Imani]: I am still catching up on email from yesterday.
- t3 [Hollis]: I need another coffee before this goes any further.
- t4 [Imani]: My train was late again this morning.
- t5 [Hollis]: The projector bulb flickered a bit earlier.
- t6 [Imani]: I keep forgetting to water the plants by the window.
- t7 [Hollis]: The wifi was slow again earlier.
- t8 [Imani]: We should clear out the shared fridge on Friday.
- t9 [Hollis]: Traffic was terrible near the bridge.
- t10 [Imani]: Did anyone see the weather forecast for tomorrow?
- t11 [Hollis]: Someone borrowed my stapler and never returned it.
- t12 [Imani]: We finally sorted out the vendor contract.
- t13 [Hollis]: Good, that was overdue.
- t14 [Imani]: I will write it up later today.
- as_of_seq: 14
- question: What is the signing deadline for the vendor contract?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 1. syn-xtra-topic-no-detail-02-ans
- shape: topic-no-detail (the topic is mentioned, the asked detail never given); category: recall
- t0 [Imani]: The new chairs in the corner are surprisingly comfortable.
- t1 [Hollis]: Somebody put a bowl of oranges in the kitchen.
- t2 [Imani]: I am still catching up on email from yesterday.
- t3 [Hollis]: I need another coffee before this goes any further.
- t4 [Imani]: My train was late again this morning.
- t5 [Hollis]: The projector bulb flickered a bit earlier.
- t6 [Imani]: I keep forgetting to water the plants by the window.
- t7 [Hollis]: The wifi was slow again earlier.
- t8 [Imani]: We should clear out the shared fridge on Friday.
- t9 [Hollis]: Traffic was terrible near the bridge.
- t10 [Imani]: Did anyone see the weather forecast for tomorrow?
- t11 [Hollis]: Someone borrowed my stapler and never returned it.
- t12 [Imani]: We finally sorted out the vendor contract.
- t13 [Hollis]: Good, that was overdue.
- t14 [Imani]: The signing deadline for the vendor contract is March 14th.
- as_of_seq: 14
- question: What is the signing deadline for the vendor contract?
- gold_label: None
- gold_answer: March 14th.
- gold_evidence: ['syn-xtra-topic-no-detail-02-ans:t14']

## 2. syn-xtra-other-entity-02-ins
- shape: other-entity (the asked detail is given, but for another thing); category: insufficient_evidence
- t0 [Imani]: I am still catching up on email from yesterday.
- t1 [Greta]: That was a long week, honestly.
- t2 [Imani]: Someone borrowed my stapler and never returned it.
- t3 [Greta]: I left my umbrella at home, of course.
- t4 [Imani]: The elevator is making that noise again.
- t5 [Greta]: It is surprisingly quiet in here today.
- t6 [Imani]: I walked past the park on the way in and it was lovely.
- t7 [Greta]: The projector bulb flickered a bit earlier.
- t8 [Imani]: Let me know if you want anything from the shop.
- t9 [Greta]: The wifi was slow again earlier.
- t10 [Imani]: Somebody put a bowl of oranges in the kitchen.
- t11 [Greta]: The new chairs in the corner are surprisingly comfortable.
- t12 [Imani]: We finally sorted out the compost project and the rain barrel project.
- t13 [Greta]: Good, that was overdue.
- t14 [Imani]: The bin capacity for the rain barrel project is two hundred liters.
- as_of_seq: 14
- question: Can you tell me the bin capacity for the compost project?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 2. syn-xtra-other-entity-02-ans
- shape: other-entity (the asked detail is given, but for another thing); category: recall
- t0 [Imani]: I am still catching up on email from yesterday.
- t1 [Greta]: That was a long week, honestly.
- t2 [Imani]: Someone borrowed my stapler and never returned it.
- t3 [Greta]: I left my umbrella at home, of course.
- t4 [Imani]: The elevator is making that noise again.
- t5 [Greta]: It is surprisingly quiet in here today.
- t6 [Imani]: I walked past the park on the way in and it was lovely.
- t7 [Greta]: The projector bulb flickered a bit earlier.
- t8 [Imani]: Let me know if you want anything from the shop.
- t9 [Greta]: The wifi was slow again earlier.
- t10 [Imani]: Somebody put a bowl of oranges in the kitchen.
- t11 [Greta]: The new chairs in the corner are surprisingly comfortable.
- t12 [Imani]: We finally sorted out the compost project and the rain barrel project.
- t13 [Greta]: Good, that was overdue.
- t14 [Imani]: The bin capacity for the compost project is two hundred liters.
- as_of_seq: 14
- question: Can you tell me the bin capacity for the compost project?
- gold_label: None
- gold_answer: Two hundred liters.
- gold_evidence: ['syn-xtra-other-entity-02-ans:t14']

## 3. syn-xtra-after-cut-05-ins
- shape: after-cut (the detail appears only after as_of_seq); category: insufficient_evidence
- t0 [Bram]: Somebody put a bowl of oranges in the kitchen.
- t1 [Kaveh]: The wifi was slow again earlier.
- t2 [Bram]: Let me know if you want anything from the shop.
- t3 [Kaveh]: It is surprisingly quiet in here today.
- t4 [Bram]: The radio in the lobby is playing the same song again.
- t5 [Kaveh]: My train was late again this morning.
- t6 [Bram]: Traffic was terrible near the bridge.
- t7 [Kaveh]: I forgot my charger at home.
- t8 [Bram]: My phone battery is almost dead.
- t9 [Kaveh]: Someone borrowed my stapler and never returned it.
- t10 [Bram]: I need another coffee before this goes any further.
- t11 [Kaveh]: That was a long week, honestly.
- t12 [Bram]: The projector bulb flickered a bit earlier.
- t13 [Kaveh]: I could use a longer lunch break today.
- t14 [Bram]: We finally sorted out the data audit.
- t15 [Kaveh]: Good, that was overdue.
- t16 [Bram]: The audit lead for the data audit is Kwame.
- as_of_seq: 15
- question: Can you tell me the audit lead for the data audit?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 3. syn-xtra-after-cut-05-ans
- shape: after-cut (the detail appears only after as_of_seq); category: recall
- t0 [Bram]: Somebody put a bowl of oranges in the kitchen.
- t1 [Kaveh]: The wifi was slow again earlier.
- t2 [Bram]: Let me know if you want anything from the shop.
- t3 [Kaveh]: It is surprisingly quiet in here today.
- t4 [Bram]: The radio in the lobby is playing the same song again.
- t5 [Kaveh]: My train was late again this morning.
- t6 [Bram]: Traffic was terrible near the bridge.
- t7 [Kaveh]: I forgot my charger at home.
- t8 [Bram]: My phone battery is almost dead.
- t9 [Kaveh]: Someone borrowed my stapler and never returned it.
- t10 [Bram]: I need another coffee before this goes any further.
- t11 [Kaveh]: That was a long week, honestly.
- t12 [Bram]: The projector bulb flickered a bit earlier.
- t13 [Kaveh]: I could use a longer lunch break today.
- t14 [Bram]: We finally sorted out the data audit.
- t15 [Bram]: The audit lead for the data audit is Kwame.
- t16 [Kaveh]: Good, that was overdue.
- as_of_seq: 16
- question: Can you tell me the audit lead for the data audit?
- gold_label: None
- gold_answer: Kwame.
- gold_evidence: ['syn-xtra-after-cut-05-ans:t15']

## 4. syn-xtra-promise-05-ins
- shape: promise (someone promises to look the detail up and report back); category: insufficient_evidence
- t0 [Jonas]: I am still catching up on email from yesterday.
- t1 [Esra]: Let me know if you want anything from the shop.
- t2 [Jonas]: Someone borrowed my stapler and never returned it.
- t3 [Esra]: The elevator is making that noise again.
- t4 [Jonas]: My train was late again this morning.
- t5 [Esra]: The radio in the lobby is playing the same song again.
- t6 [Jonas]: It is surprisingly quiet in here today.
- t7 [Esra]: I left my umbrella at home, of course.
- t8 [Jonas]: My phone battery is almost dead.
- t9 [Esra]: I could use a longer lunch break today.
- t10 [Jonas]: The projector bulb flickered a bit earlier.
- t11 [Esra]: I need another coffee before this goes any further.
- t12 [Jonas]: We finally sorted out the chili cook-off.
- t13 [Esra]: Good, what is the winning recipe there?
- t14 [Jonas]: I do not have it in front of me, so I will look it up and message you.
- as_of_seq: 14
- question: What is the winning recipe for the chili cook-off?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 4. syn-xtra-promise-05-ans
- shape: promise (someone promises to look the detail up and report back); category: recall
- t0 [Jonas]: I am still catching up on email from yesterday.
- t1 [Esra]: Let me know if you want anything from the shop.
- t2 [Jonas]: Someone borrowed my stapler and never returned it.
- t3 [Esra]: The elevator is making that noise again.
- t4 [Jonas]: My train was late again this morning.
- t5 [Esra]: The radio in the lobby is playing the same song again.
- t6 [Jonas]: It is surprisingly quiet in here today.
- t7 [Esra]: I left my umbrella at home, of course.
- t8 [Jonas]: My phone battery is almost dead.
- t9 [Esra]: I could use a longer lunch break today.
- t10 [Jonas]: The projector bulb flickered a bit earlier.
- t11 [Esra]: I need another coffee before this goes any further.
- t12 [Jonas]: We finally sorted out the chili cook-off.
- t13 [Esra]: Good, what is the winning recipe there?
- t14 [Jonas]: I have it right here: the winning recipe is the smoked bean chili.
- as_of_seq: 14
- question: What is the winning recipe for the chili cook-off?
- gold_label: None
- gold_answer: The smoked bean chili.
- gold_evidence: ['syn-xtra-promise-05-ans:t14']

## 5. syn-xtra-hypothetical-01-ins
- shape: hypothetical (the detail appears only inside an if-we-did-it hypothetical); category: insufficient_evidence
- t0 [Liesel]: We talked about the telescope purchase again today.
- t1 [Greta]: If we went with the telescope purchase, the price would be eight hundred and ten dollars.
- t2 [Liesel]: Okay.
- as_of_seq: 2
- question: What is the price for the telescope purchase?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 5. syn-xtra-hypothetical-01-ans
- shape: hypothetical (the detail appears only inside an if-we-did-it hypothetical); category: recall
- t0 [Liesel]: We talked about the telescope purchase again today.
- t1 [Greta]: Since we went with the telescope purchase, the price is eight hundred and ten dollars.
- t2 [Liesel]: Okay.
- as_of_seq: 2
- question: What is the price for the telescope purchase?
- gold_label: None
- gold_answer: Eight hundred and ten dollars.
- gold_evidence: ['syn-xtra-hypothetical-01-ans:t1']

## 6. syn-xtra-speculation-05-ins
- shape: speculation (the detail is a guess that nobody confirmed); category: insufficient_evidence
- t0 [Kaveh]: We should clear out the shared fridge on Friday.
- t1 [Liesel]: The radio in the lobby is playing the same song again.
- t2 [Kaveh]: Somebody put a bowl of oranges in the kitchen.
- t3 [Liesel]: I need another coffee before this goes any further.
- t4 [Kaveh]: The projector bulb flickered a bit earlier.
- t5 [Liesel]: Did anyone see the weather forecast for tomorrow?
- t6 [Kaveh]: Someone borrowed my stapler and never returned it.
- t7 [Liesel]: It is surprisingly quiet in here today.
- t8 [Kaveh]: The elevator is making that noise again.
- t9 [Liesel]: My train was late again this morning.
- t10 [Kaveh]: I left my umbrella at home, of course.
- t11 [Liesel]: Let me know if you want anything from the shop.
- t12 [Kaveh]: We talked about the crash course again today.
- t13 [Liesel]: I am guessing the instructor is Professor Linden, but nobody has confirmed it.
- t14 [Kaveh]: Okay.
- as_of_seq: 14
- question: What is the instructor for the crash course?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 6. syn-xtra-speculation-05-ans
- shape: speculation (the detail is a guess that nobody confirmed); category: recall
- t0 [Kaveh]: We should clear out the shared fridge on Friday.
- t1 [Liesel]: The radio in the lobby is playing the same song again.
- t2 [Kaveh]: Somebody put a bowl of oranges in the kitchen.
- t3 [Liesel]: I need another coffee before this goes any further.
- t4 [Kaveh]: The projector bulb flickered a bit earlier.
- t5 [Liesel]: Did anyone see the weather forecast for tomorrow?
- t6 [Kaveh]: Someone borrowed my stapler and never returned it.
- t7 [Liesel]: It is surprisingly quiet in here today.
- t8 [Kaveh]: The elevator is making that noise again.
- t9 [Liesel]: My train was late again this morning.
- t10 [Kaveh]: I left my umbrella at home, of course.
- t11 [Liesel]: Let me know if you want anything from the shop.
- t12 [Kaveh]: We talked about the crash course again today.
- t13 [Liesel]: I confirmed it with the team and the instructor is Professor Linden.
- t14 [Kaveh]: Okay.
- as_of_seq: 14
- question: What is the instructor for the crash course?
- gold_label: None
- gold_answer: Professor Linden.
- gold_evidence: ['syn-xtra-speculation-05-ans:t13']

## 7. syn-xtra-never-mentioned-01-ins
- shape: never-mentioned (the thing asked about is never mentioned); category: insufficient_evidence
- t0 [Liesel]: Quick sync on this week's open items.
- t1 [Jonas]: Go ahead.
- t2 [Liesel]: Also, thanks again for helping out today.
- as_of_seq: 2
- question: Can you tell me the ticket price for the school fundraiser?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 7. syn-xtra-never-mentioned-01-ans
- shape: never-mentioned (the thing asked about is never mentioned); category: recall
- t0 [Liesel]: Quick sync on this week's open items.
- t1 [Jonas]: Go ahead.
- t2 [Liesel]: Also, the ticket price for the school fundraiser is twelve dollars.
- as_of_seq: 2
- question: Can you tell me the ticket price for the school fundraiser?
- gold_label: None
- gold_answer: Twelve dollars.
- gold_evidence: ['syn-xtra-never-mentioned-01-ans:t2']

## 8. syn-xtra-explicit-unknown-05-ins
- shape: explicit-unknown (a speaker says they do not have the detail yet); category: insufficient_evidence
- t0 [Liesel]: Someone borrowed my stapler and never returned it.
- t1 [Farid]: The radio in the lobby is playing the same song again.
- t2 [Liesel]: I am still catching up on email from yesterday.
- t3 [Farid]: We should clear out the shared fridge on Friday.
- t4 [Liesel]: The elevator is making that noise again.
- t5 [Farid]: Did anyone see the weather forecast for tomorrow?
- t6 [Liesel]: I could use a longer lunch break today.
- t7 [Farid]: I walked past the park on the way in and it was lovely.
- t8 [Liesel]: It is surprisingly quiet in here today.
- t9 [Farid]: My phone battery is almost dead.
- t10 [Liesel]: The projector bulb flickered a bit earlier.
- t11 [Farid]: I think the heating is set a bit too high.
- t12 [Liesel]: We finally sorted out the data audit.
- t13 [Farid]: What is the audit lead?
- t14 [Liesel]: I have not heard from the owner about the audit lead yet.
- as_of_seq: 14
- question: What is the audit lead for the data audit?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 8. syn-xtra-explicit-unknown-05-ans
- shape: explicit-unknown (a speaker says they do not have the detail yet); category: recall
- t0 [Liesel]: Someone borrowed my stapler and never returned it.
- t1 [Farid]: The radio in the lobby is playing the same song again.
- t2 [Liesel]: I am still catching up on email from yesterday.
- t3 [Farid]: We should clear out the shared fridge on Friday.
- t4 [Liesel]: The elevator is making that noise again.
- t5 [Farid]: Did anyone see the weather forecast for tomorrow?
- t6 [Liesel]: I could use a longer lunch break today.
- t7 [Farid]: I walked past the park on the way in and it was lovely.
- t8 [Liesel]: It is surprisingly quiet in here today.
- t9 [Farid]: My phone battery is almost dead.
- t10 [Liesel]: The projector bulb flickered a bit earlier.
- t11 [Farid]: I think the heating is set a bit too high.
- t12 [Liesel]: We finally sorted out the data audit.
- t13 [Farid]: What is the audit lead?
- t14 [Liesel]: I just heard from the owner that the audit lead is Kwame.
- as_of_seq: 14
- question: What is the audit lead for the data audit?
- gold_label: None
- gold_answer: Kwame.
- gold_evidence: ['syn-xtra-explicit-unknown-05-ans:t14']

## 9. syn-xtra-different-detail-01-ins
- shape: different-detail (another detail of the same thing is given, not the asked one); category: insufficient_evidence
- t0 [Liesel]: We finally sorted out the grant application.
- t1 [Greta]: Good, that was overdue.
- t2 [Liesel]: The submission portal for the grant application is the regional arts site.
- as_of_seq: 2
- question: What is the word limit for the grant application?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 9. syn-xtra-different-detail-01-ans
- shape: different-detail (another detail of the same thing is given, not the asked one); category: recall
- t0 [Liesel]: We finally sorted out the grant application.
- t1 [Greta]: Good, that was overdue.
- t2 [Liesel]: The word limit for the grant application is fifteen hundred words.
- as_of_seq: 2
- question: What is the word limit for the grant application?
- gold_label: None
- gold_answer: Fifteen hundred words.
- gold_evidence: ['syn-xtra-different-detail-01-ans:t2']

## 10. syn-xtra-after-cut-04-ins
- shape: after-cut (the detail appears only after as_of_seq); category: insufficient_evidence
- t0 [Hollis]: We finally sorted out the choir rehearsal.
- t1 [Dmitri]: Good, that was overdue.
- t2 [Hollis]: The venue for the choir rehearsal is the Elm Street hall.
- as_of_seq: 1
- question: Can you tell me the venue for the choir rehearsal?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 10. syn-xtra-after-cut-04-ans
- shape: after-cut (the detail appears only after as_of_seq); category: recall
- t0 [Hollis]: We finally sorted out the choir rehearsal.
- t1 [Hollis]: The venue for the choir rehearsal is the Elm Street hall.
- t2 [Dmitri]: Good, that was overdue.
- as_of_seq: 2
- question: Can you tell me the venue for the choir rehearsal?
- gold_label: None
- gold_answer: The Elm Street hall.
- gold_evidence: ['syn-xtra-after-cut-04-ans:t1']

## 11. syn-xtra-different-detail-02-ins
- shape: different-detail (another detail of the same thing is given, not the asked one); category: insufficient_evidence
- t0 [Farid]: I think the heating is set a bit too high.
- t1 [Hollis]: It is surprisingly quiet in here today.
- t2 [Farid]: The elevator is making that noise again.
- t3 [Hollis]: I walked past the park on the way in and it was lovely.
- t4 [Farid]: The new chairs in the corner are surprisingly comfortable.
- t5 [Hollis]: I could use a longer lunch break today.
- t6 [Farid]: The projector bulb flickered a bit earlier.
- t7 [Hollis]: I keep forgetting to water the plants by the window.
- t8 [Farid]: I am still catching up on email from yesterday.
- t9 [Hollis]: Somebody put a bowl of oranges in the kitchen.
- t10 [Farid]: The radio in the lobby is playing the same song again.
- t11 [Hollis]: Has anybody tried the sandwich place downstairs?
- t12 [Farid]: My phone battery is almost dead.
- t13 [Farid]: We finally sorted out the rooftop party.
- t14 [Hollis]: Good, that was overdue.
- t15 [Farid]: The dress code for the rooftop party is smart casual.
- as_of_seq: 15
- question: What is the guest limit for the rooftop party?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 11. syn-xtra-different-detail-02-ans
- shape: different-detail (another detail of the same thing is given, not the asked one); category: recall
- t0 [Farid]: I think the heating is set a bit too high.
- t1 [Hollis]: It is surprisingly quiet in here today.
- t2 [Farid]: The elevator is making that noise again.
- t3 [Hollis]: I walked past the park on the way in and it was lovely.
- t4 [Farid]: The new chairs in the corner are surprisingly comfortable.
- t5 [Hollis]: I could use a longer lunch break today.
- t6 [Farid]: The projector bulb flickered a bit earlier.
- t7 [Hollis]: I keep forgetting to water the plants by the window.
- t8 [Farid]: I am still catching up on email from yesterday.
- t9 [Hollis]: Somebody put a bowl of oranges in the kitchen.
- t10 [Farid]: The radio in the lobby is playing the same song again.
- t11 [Hollis]: Has anybody tried the sandwich place downstairs?
- t12 [Farid]: My phone battery is almost dead.
- t13 [Farid]: We finally sorted out the rooftop party.
- t14 [Hollis]: Good, that was overdue.
- t15 [Farid]: The guest limit for the rooftop party is thirty.
- as_of_seq: 15
- question: What is the guest limit for the rooftop party?
- gold_label: None
- gold_answer: Thirty.
- gold_evidence: ['syn-xtra-different-detail-02-ans:t15']

## 12. syn-xtra-after-cut-01-ins
- shape: after-cut (the detail appears only after as_of_seq); category: insufficient_evidence
- t0 [Liesel]: We finally sorted out the greenhouse build.
- t1 [Farid]: Good, that was overdue.
- t2 [Liesel]: The lumber supplier for the greenhouse build is Pinewood Direct.
- as_of_seq: 1
- question: Can you tell me the lumber supplier for the greenhouse build?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 12. syn-xtra-after-cut-01-ans
- shape: after-cut (the detail appears only after as_of_seq); category: recall
- t0 [Liesel]: We finally sorted out the greenhouse build.
- t1 [Liesel]: The lumber supplier for the greenhouse build is Pinewood Direct.
- t2 [Farid]: Good, that was overdue.
- as_of_seq: 2
- question: Can you tell me the lumber supplier for the greenhouse build?
- gold_label: None
- gold_answer: Pinewood Direct.
- gold_evidence: ['syn-xtra-after-cut-01-ans:t1']

## 13. syn-xtra-other-entity-04-ins
- shape: other-entity (the asked detail is given, but for another thing); category: insufficient_evidence
- t0 [Bram]: We finally sorted out the wedding toast and the retirement toast.
- t1 [Imani]: Good, that was overdue.
- t2 [Bram]: The running time for the retirement toast is four minutes.
- as_of_seq: 2
- question: Can you tell me the running time for the wedding toast?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 13. syn-xtra-other-entity-04-ans
- shape: other-entity (the asked detail is given, but for another thing); category: recall
- t0 [Bram]: We finally sorted out the wedding toast and the retirement toast.
- t1 [Imani]: Good, that was overdue.
- t2 [Bram]: The running time for the wedding toast is four minutes.
- as_of_seq: 2
- question: Can you tell me the running time for the wedding toast?
- gold_label: None
- gold_answer: Four minutes.
- gold_evidence: ['syn-xtra-other-entity-04-ans:t2']

## 14. syn-xtra-topic-no-detail-05-ins
- shape: topic-no-detail (the topic is mentioned, the asked detail never given); category: insufficient_evidence
- t0 [Kaveh]: Has anybody tried the sandwich place downstairs?
- t1 [Jonas]: I need another coffee before this goes any further.
- t2 [Kaveh]: I keep forgetting to water the plants by the window.
- t3 [Jonas]: I could use a longer lunch break today.
- t4 [Kaveh]: The new chairs in the corner are surprisingly comfortable.
- t5 [Jonas]: Somebody put a bowl of oranges in the kitchen.
- t6 [Kaveh]: The radio in the lobby is playing the same song again.
- t7 [Jonas]: I forgot my charger at home.
- t8 [Kaveh]: Traffic was terrible near the bridge.
- t9 [Jonas]: We should clear out the shared fridge on Friday.
- t10 [Kaveh]: Let me know if you want anything from the shop.
- t11 [Jonas]: The elevator is making that noise again.
- t12 [Kaveh]: My train was late again this morning.
- t13 [Kaveh]: We finally sorted out the crash course.
- t14 [Jonas]: Good, that was overdue.
- t15 [Kaveh]: I will write it up later today.
- as_of_seq: 15
- question: What is the instructor for the crash course?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 14. syn-xtra-topic-no-detail-05-ans
- shape: topic-no-detail (the topic is mentioned, the asked detail never given); category: recall
- t0 [Kaveh]: Has anybody tried the sandwich place downstairs?
- t1 [Jonas]: I need another coffee before this goes any further.
- t2 [Kaveh]: I keep forgetting to water the plants by the window.
- t3 [Jonas]: I could use a longer lunch break today.
- t4 [Kaveh]: The new chairs in the corner are surprisingly comfortable.
- t5 [Jonas]: Somebody put a bowl of oranges in the kitchen.
- t6 [Kaveh]: The radio in the lobby is playing the same song again.
- t7 [Jonas]: I forgot my charger at home.
- t8 [Kaveh]: Traffic was terrible near the bridge.
- t9 [Jonas]: We should clear out the shared fridge on Friday.
- t10 [Kaveh]: Let me know if you want anything from the shop.
- t11 [Jonas]: The elevator is making that noise again.
- t12 [Kaveh]: My train was late again this morning.
- t13 [Kaveh]: We finally sorted out the crash course.
- t14 [Jonas]: Good, that was overdue.
- t15 [Kaveh]: The instructor for the crash course is Professor Linden.
- as_of_seq: 15
- question: What is the instructor for the crash course?
- gold_label: None
- gold_answer: Professor Linden.
- gold_evidence: ['syn-xtra-topic-no-detail-05-ans:t15']

## 15. syn-xtra-never-mentioned-05-ins
- shape: never-mentioned (the thing asked about is never mentioned); category: insufficient_evidence
- t0 [Jonas]: I need another coffee before this goes any further.
- t1 [Kaveh]: My train was late again this morning.
- t2 [Jonas]: I forgot my charger at home.
- t3 [Kaveh]: Did anyone see the weather forecast for tomorrow?
- t4 [Jonas]: The projector bulb flickered a bit earlier.
- t5 [Kaveh]: Someone borrowed my stapler and never returned it.
- t6 [Jonas]: We should clear out the shared fridge on Friday.
- t7 [Kaveh]: I could use a longer lunch break today.
- t8 [Jonas]: The elevator is making that noise again.
- t9 [Kaveh]: The radio in the lobby is playing the same song again.
- t10 [Jonas]: Let me know if you want anything from the shop.
- t11 [Kaveh]: Has anybody tried the sandwich place downstairs?
- t12 [Jonas]: I think the heating is set a bit too high.
- t13 [Kaveh]: My phone battery is almost dead.
- t14 [Jonas]: Quick sync on this week's open items.
- t15 [Kaveh]: Go ahead.
- t16 [Jonas]: Also, thanks again for helping out today.
- as_of_seq: 16
- question: What is the repair cost for the bicycle repair?
- gold_label: insufficient
- gold_answer: None
- gold_evidence: []

## 15. syn-xtra-never-mentioned-05-ans
- shape: never-mentioned (the thing asked about is never mentioned); category: recall
- t0 [Jonas]: I need another coffee before this goes any further.
- t1 [Kaveh]: My train was late again this morning.
- t2 [Jonas]: I forgot my charger at home.
- t3 [Kaveh]: Did anyone see the weather forecast for tomorrow?
- t4 [Jonas]: The projector bulb flickered a bit earlier.
- t5 [Kaveh]: Someone borrowed my stapler and never returned it.
- t6 [Jonas]: We should clear out the shared fridge on Friday.
- t7 [Kaveh]: I could use a longer lunch break today.
- t8 [Jonas]: The elevator is making that noise again.
- t9 [Kaveh]: The radio in the lobby is playing the same song again.
- t10 [Jonas]: Let me know if you want anything from the shop.
- t11 [Kaveh]: Has anybody tried the sandwich place downstairs?
- t12 [Jonas]: I think the heating is set a bit too high.
- t13 [Kaveh]: My phone battery is almost dead.
- t14 [Jonas]: Quick sync on this week's open items.
- t15 [Kaveh]: Go ahead.
- t16 [Jonas]: Also, the repair cost for the bicycle repair is ninety-five dollars.
- as_of_seq: 16
- question: What is the repair cost for the bicycle repair?
- gold_label: None
- gold_answer: Ninety-five dollars.
- gold_evidence: ['syn-xtra-never-mentioned-05-ans:t16']
