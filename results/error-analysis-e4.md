# E4 error analysis: the trained model's mistakes on dev

547 mistakes among 1082 scored dev cases; 50 are listed, chosen by a fixed rule (categories in alphabetical order, one at a time from each, by id). DECODE cases appear by id only.

## Mistakes by category

- contradiction: 17
- decode: 497
- insufficient_evidence: 16
- legitimate_update: 17

## Listed mistakes

### syn-con-00 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Maya: I've never been to Lisbon.
- Dev: You'd love the tiles there.
- statement, Maya: Last spring in Lisbon I got lost looking for the tram stop.

### decode-dev-0010abb15e54 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-00 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.00
- Maya: I'll bring the projector on Thursday.
- statement, Dev: Great, I'll set up the room.

### syn-deadline-update-00 (legitimate_update)
- gold: update; predicted: consistent, probability 0.00
- Ravi: The deadline is Monday
- statement, Sue: We moved the deadline to Wednesday

### syn-con-02 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Priya: I'm allergic to peanuts, so no satay for me.
- Sam: Noted, we'll pick another place.
- statement, Priya: The peanut satay was great, I had two helpings last week.

### decode-dev-00383dc41f9b (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-02 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.00
- Priya: Sam will review the contract.
- statement, Sam: I'll get to it soon.

### syn-upd-00 (legitimate_update)
- gold: update; predicted: consistent, probability 0.00
- Maya: The team lunch is at noon.
- Dev: Works for me.
- statement, Maya: Change of plan, the team lunch is now at one.

### syn-con-04 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Ines: I don't drive, I never got a licence.
- Jo: Same here, I take the bus.
- statement, Ines: I drove the rental van down to the coast on my own.

### decode-dev-003df04543c9 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-04 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.00
- Ines: I talked to the client yesterday.
- Jo: What did they say?
- statement, Ines: Mostly small talk.

### syn-upd-02 (legitimate_update)
- gold: update; predicted: consistent, probability 0.00
- Priya: The venue holds fifty people.
- Sam: Plenty.
- statement, Priya: The venue just told us they expanded the hall, it holds eighty now.

### syn-con-06 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Nora: I haven't spoken to the client since March.
- Ben: They keep asking about you.
- statement, Nora: I talked to the client yesterday and they were happy with the draft.

### decode-dev-007ff8f1af0a (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-06 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.00
- Nora: Ben is going to the conference.
- statement, Ben: Yes, looking forward to it.

### syn-upd-04 (legitimate_update)
- gold: update; predicted: consistent, probability 0.00
- Ines: The client wants the blue logo.
- Jo: Easy.
- statement, Ines: Update from the client, they switched to the green logo.

### syn-con-08 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Zara: I'm the only one on the design team.
- Luis: That's a lot of work for one person.
- statement, Zara: Our three designers split the mockups between them.

### decode-dev-0081d887e869 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-08 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.00
- Zara: The prototype is ready for testing.
- statement, Luis: Excellent.

### syn-upd-06 (legitimate_update)
- gold: update; predicted: consistent, probability 0.00
- Nora: Dinner reservation is at the Thai place.
- Ben: Yum.
- statement, Nora: They had a flood, so I moved the reservation to the Italian place.

### syn-con-10 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Chloe: I've been vegetarian for ten years.
- Raj: Then the steakhouse is out.
- statement, Chloe: I had the brisket there on Saturday and it was delicious.

### decode-dev-0091020fa185 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-10 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.00
- Chloe: I have a dentist appointment Friday.
- statement, Raj: Hope it goes well.

### syn-upd-08 (legitimate_update)
- gold: update; predicted: consistent, probability 0.00
- Zara: Luis is leading the migration.
- Luis: Yes, I'm on it.
- statement, Zara: Luis moved to another project, so Mei is leading the migration now.

### syn-con-12 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Amara: Nobody on my team works weekends.
- Pete: Healthy approach.
- statement, Amara: My whole team was in on Saturday and Sunday to finish the migration.

### decode-dev-009c81ba7302 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-12 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.00
- Amara: We'll discuss pricing next week.
- statement, Pete: Sure.

### syn-upd-10 (legitimate_update)
- gold: update; predicted: consistent, probability 0.00
- Chloe: The password policy needs eight characters.
- Raj: Fine.
- statement, Chloe: Security updated the policy, it requires twelve characters now.

### syn-con-14 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Bea: The lease is for two years, signed last week.
- Carl: Congrats on the new place.
- statement, Bea: I haven't signed anything yet, the lease is still with the lawyer.

### decode-dev-00dbb4269c1b (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-14 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.00
- Bea: I'm moving apartments soon.
- statement, Carl: Exciting.

### syn-upd-12 (legitimate_update)
- gold: update; predicted: consistent, probability 0.00
- Amara: I'm vegetarian these days.
- Pete: I'll order accordingly.
- statement, Amara: Actually I started eating fish again last month, so pescatarian now.

### syn-con-16 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Gus: I grew up in Texas and never left until university.
- Hana: Wow, that's a long time in one place.
- statement, Gus: I spent my childhood in Norway before we ever came to the States.

### decode-dev-00e34b34bc40 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-16 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.00
- Gus: I read the paper last night.
- Hana: And?
- statement, Gus: It was interesting.

### syn-upd-14 (legitimate_update)
- gold: update; predicted: consistent, probability 0.00
- Bea: Anna is our point of contact.
- Carl: Good to know.
- statement, Bea: Anna went on leave, so Daniel is the new point of contact.

### syn-con-18 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Kai: There are only three people coming to the offsite.
- Lia: A small group then.
- statement, Kai: I'm booking forty seats for the offsite dinner, since everyone is coming.

### decode-dev-012cb59a23f2 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-18 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.00
- Kai: We are doing a survey.
- statement, Lia: Good idea.

### syn-upd-16 (legitimate_update)
- gold: update; predicted: consistent, probability 0.00
- Gus: The ticket price is 20 euros.
- Hana: Fair.
- statement, Gus: They put a student discount in place, so students pay 12 euros.

### syn-con-20 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Otto: My sister is younger than me.
- Pia: How much younger?
- statement, Otto: My sister is the oldest of us, she's got six years on me.

### decode-dev-012f68f0649b (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-20 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.00
- Otto: I bought a new bike.
- statement, Pia: Congrats.

### syn-upd-18 (legitimate_update)
- gold: update; predicted: consistent, probability 0.00
- Kai: I'm on the early shift this week.
- Lia: Right.
- statement, Kai: I swapped with Lia, so I'm on the late shift this week.

### syn-con-22 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Stan: The printer on the second floor works fine.
- Tess: Good.
- statement, Stan: Nobody can print upstairs, that printer has been dead for a month.

### decode-dev-014430684767 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file

### syn-ins-22 (insufficient_evidence)
- gold: insufficient; predicted: consistent, probability 0.00
- Stan: We're migrating to a new database.
- statement, Tess: About time.

### syn-upd-20 (legitimate_update)
- gold: update; predicted: consistent, probability 0.00
- Otto: I'm planning to buy the red bike.
- Pia: Nice colour.
- statement, Otto: I changed my mind and I'm going with the black one.

### syn-con-24 (contradiction)
- gold: contradiction; predicted: consistent, probability 0.00
- Walt: The patch was tested on all three platforms.
- Xena: Thorough.
- statement, Walt: I only had time to test the patch on Linux, the others are untested.

### decode-dev-014cfcd255b9 (decode)
- gold: contradiction; predicted: consistent
- text not shown (DECODE); look the id up in the local file
