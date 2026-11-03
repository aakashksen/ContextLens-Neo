# E4 error analysis: the trained model's mistakes on dev

101 mistakes among 1082 scored dev cases; 50 are listed, chosen by a fixed rule (categories in alphabetical order, one at a time from each, by id). DECODE's score covers contradiction vs not only and is not four-label performance; its cases appear here by id only.

## Mistakes by category

- contradiction: 5
- decode: 83
- hypothetical_or_joking: 1
- insufficient_evidence: 4
- legitimate_update: 1
- wrong_speaker_trap: 7

## Listed mistakes

### syn-con-00 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.92
- Maya: I've never been to Lisbon.
- Dev: You'd love the tiles there.
- statement, Maya: Last spring in Lisbon I got lost looking for the tram stop.

### decode-dev-009377a15b7f (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### syn-hyp-20 (hypothetical_or_joking)
- gold: consistent; predicted: contradiction, probability 0.97
- Otto: I own one car.
- Pia: Okay.
- statement, Otto: If I owned ten cars I'd still be late, that's the joke.

### syn-ins-02 (insufficient_evidence)
- gold: insufficient; predicted: update, probability 1.00
- Priya: Sam will review the contract.
- statement, Sam: I'll get to it soon.

### syn-upd-12 (legitimate_update)
- gold: update; predicted: consistent, probability 1.00
- Amara: I'm vegetarian these days.
- Pete: I'll order accordingly.
- statement, Amara: Actually I started eating fish again last month, so pescatarian now.

### syn-wsc-08 (wrong_speaker_trap)
- gold: consistent; predicted: contradiction, probability 0.92
- Zara: I live on the north side.
- Luis: Okay.
- statement, Luis: I live on the south side.

### syn-con-04 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.96
- Ines: I don't drive, I never got a licence.
- Jo: Same here, I take the bus.
- statement, Ines: I drove the rental van down to the coast on my own.

### decode-dev-012cb59a23f2 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-04 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 1.00
- Ines: I talked to the client yesterday.
- Jo: What did they say?
- statement, Ines: Mostly small talk.

### syn-wsc-14 (wrong_speaker_trap)
- gold: consistent; predicted: contradiction, probability 1.00
- Carl: Our house has a garden.
- Dana: Lovely.
- statement, Dana: Our flat has no garden at all.

### syn-con-10 (contradiction)
- gold: contradiction; predicted: consistent, probability 1.00
- Chloe: I've been vegetarian for ten years.
- Raj: Then the steakhouse is out.
- statement, Chloe: I had the brisket there on Saturday and it was delicious.

### decode-dev-013809d0ec5b (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### syn-ins-06 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.83
- Nora: Ben is going to the conference.
- statement, Ben: Yes, looking forward to it.

### syn-wsc-18 (wrong_speaker_trap)
- gold: consistent; predicted: contradiction, probability 1.00
- Lia: I voted yes on the proposal.
- Mona: Okay.
- statement, Mona: I voted no on the proposal.

### syn-con-12 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.65
- Amara: Nobody on my team works weekends.
- Pete: Healthy approach.
- statement, Amara: My whole team was in on Saturday and Sunday to finish the migration.

### decode-dev-022315ac15a9 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### syn-ins-16 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.98
- Gus: I read the paper last night.
- Hana: And?
- statement, Gus: It was interesting.

### syn-wsc-20 (wrong_speaker_trap)
- gold: consistent; predicted: contradiction, probability 1.00
- Pia: I've been to Peru.
- Quinn: Wow.
- statement, Quinn: I've never been to Peru.

### syn-deadline-contradiction-00 (contradiction)
- gold: contradiction; predicted: update, probability 0.99
- Ravi: The deadline is Monday
- Sue: We moved the deadline to Wednesday
- statement, Sue: It has always been Wednesday

### decode-dev-02de0d40699c (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### syn-wsc-24 (wrong_speaker_trap)
- gold: consistent; predicted: contradiction, probability 1.00
- Xena: I'm ready to present.
- Yara: Good.
- statement, Yara: I'm not ready to present yet.

### decode-dev-034d3b4523b7 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### syn-wsc-26 (wrong_speaker_trap)
- gold: consistent; predicted: contradiction, probability 0.96
- Bria: I own a dog.
- Cleo: Aww.
- statement, Cleo: I own two cats, no dog.

### decode-dev-0402389ecc90 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### syn-wsc-30 (wrong_speaker_trap)
- gold: consistent; predicted: contradiction, probability 0.87
- Kim: I pay by card.
- Leo: Okay.
- statement, Leo: I pay in cash only.

### decode-dev-05d78f5b365b (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-0698b588db73 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### decode-dev-076c7d1fa4ab (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-086111c5de65 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### decode-dev-08d14a8b38f4 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-0c1f0f2f250b (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### decode-dev-0f473c47d336 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-0f63cec10a65 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-0f9e92d39b8f (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### decode-dev-10cfd7ea42e8 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-10d818597be1 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### decode-dev-1127f0b23f7a (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### decode-dev-11687bdecff9 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-11d6fe91a459 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-123b8bb80033 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-12ba7f405c54 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-12c2d2c3db2e (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-131e97879caf (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-13368982e28f (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-1444776a901a (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-16a3ac1d547d (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-17ed959d57a4 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### decode-dev-189c0b508081 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-18a529f2ebfd (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file

### decode-dev-1954815f20b9 (decode)
- gold: consistent; predicted: contradiction
- text not shown (DECODE); look the id up in the local file
