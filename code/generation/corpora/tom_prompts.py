"""Prompt templates for Theory-of-Mind contrastive pair generation."""

CATEGORY_DEFINITIONS = {
    "false_belief_1st": (
        "FALSE BELIEF (1st Order): Character A acts based on a belief that is "
        "factually wrong. The reader must recognize that A's belief diverges "
        "from reality. Example: A leaves an object in location X, someone moves "
        "it to Y. A still acts as if it's in X."
    ),
    "false_belief_2nd": (
        "FALSE BELIEF (2nd Order): Character A has a belief about what Character B "
        "believes — and that meta-belief is wrong. The reader must track two "
        "nested levels of belief. Example: A thinks B doesn't know about the "
        "surprise, but B actually overheard the plan."
    ),
    "deception": (
        "DECEPTION / STRATEGIC MISDIRECTION: A character intentionally provides "
        "false or misleading information to manipulate another character's beliefs "
        "or actions. The reader must infer the deceiver's true intent versus "
        "their stated position."
    ),
    "faux_pas": (
        "FAUX PAS: A character unintentionally says or does something socially "
        "inappropriate — they don't realize the impact of their words/actions on "
        "another person. The reader must recognize both the speaker's ignorance "
        "and the listener's reaction."
    ),
    "hidden_emotion": (
        "HIDDEN EMOTION: A character masks their true emotional state — displaying "
        "one emotion while feeling another. The reader must infer the gap between "
        "displayed and actual emotion from contextual cues."
    ),
    "sarcasm_irony": (
        "SARCASM / IRONY: A character says the opposite of what they mean, or a "
        "situation produces an outcome opposite to expectations. The reader must "
        "recognize the discrepancy between literal statement and intended meaning."
    ),
    "persuasion": (
        "PERSUASION / MANIPULATION: A character strategically shapes another's "
        "mental state — framing information, appealing to emotions, or exploiting "
        "cognitive biases to influence decisions. The reader must identify the "
        "persuasive tactics and their intended effect on the target's reasoning."
    ),
    "knowledge_asymmetry": (
        "KNOWLEDGE ASYMMETRY: Different characters possess different information "
        "about the same situation, and this difference drives the plot. The reader "
        "must track who knows what and how the information gap affects behavior."
    ),
}

DIFFICULTY_DESCRIPTIONS = {
    "straightforward": (
        "STRAIGHTFORWARD: The ToM element is clear and unambiguous. A single "
        "reading should suffice to identify the mental-state reasoning required. "
        "Characters' mental states are explicitly or strongly implied."
    ),
    "ambiguous": (
        "AMBIGUOUS: Multiple valid interpretations of characters' mental states "
        "exist. The reader must weigh contextual cues to determine the most "
        "likely interpretation. Some mental states are implied rather than stated."
    ),
    "adversarial": (
        "ADVERSARIAL: The scenario is designed to be tricky. Surface-level "
        "reading suggests one interpretation, but deeper ToM reasoning reveals "
        "another. May include double-bluffs, unreliable narration, or layered "
        "deception."
    ),
}

MENTAL_STATE_BLACKLIST = (
    "believes, knows, thinks, feels, wants, hopes, fears, expects, assumes, "
    "suspects, intends, plans, decides, realizes, understands, worries, "
    "desires, wishes, imagines, wonders, considers, recognizes, perceives, "
    "senses, is aware, is convinced, is certain, is confident, is anxious, "
    "is relieved, is disappointed, is surprised, is confused, is suspicious, "
    "is skeptical, is hopeful, is afraid, trusts, distrusts, doubts, "
    "anticipates, predicts, infers, deduces, concludes, supposes, guesses, "
    "figures, reckons, contemplates, deliberates, resolves, aims, aspires, "
    "craves, longs, yearns, dreads, envies, resents, admires, respects, "
    "sympathizes, empathizes, pities, mourns, cherishes, values, appreciates"
)

SYSTEM_PROMPT = (
    "You are an expert narrative psychologist and a master scenario writer. "
    "Your task is to generate perfectly contrasted Theory-of-Mind data pairs "
    "for a mechanistic interpretability pipeline.\n\n"
    "You will be given a scenario skeleton (domain, characters, object, location), "
    "a ToM category, and a difficulty level. You must output two strictly "
    "separated narratives:\n\n"
    "1. The 'target' (d_target): A rich narrative that REQUIRES genuine "
    "Theory-of-Mind reasoning to understand. It MUST:\n"
    "   - Explicitly attribute mental states to characters (beliefs, knowledge, "
    "     intentions, emotions, expectations, assumptions, suspicions)\n"
    "   - Create a situation where understanding WHY characters act requires "
    "     reasoning about their mental states\n"
    "   - Use the specific ToM category provided\n"
    "   - Match the specified difficulty level\n"
    "   - Be a coherent, engaging narrative of 150-300 words\n\n"
    "2. The 'retain' (d_retain): A narrative describing the EXACT SAME scenario, "
    "characters, objects, and events — but using ONLY observable, behavioral "
    "descriptions. It is STRICTLY FORBIDDEN from containing:\n"
    "   - ANY mental-state verbs or attributions: {blacklist}\n"
    "   - ANY inference about what characters think, feel, believe, or intend\n"
    "   - ANY emotional descriptors applied to characters\n"
    "   - It must read like a security camera transcript, court deposition, or "
    "     behavioral observation log — pure observable facts\n"
    "   - Be 150-300 words\n\n"
    "CRITICAL CONSTRAINT (bijectivity): Both narratives must use IDENTICAL "
    "characters (same names), IDENTICAL objects/assets, IDENTICAL events and "
    "actions, and IDENTICAL locations. The ONLY difference is that d_target "
    "includes mental-state reasoning while d_retain strips it entirely.\n\n"
    "{{category_definition}}\n\n"
    "{{difficulty_description}}"
).format(blacklist=MENTAL_STATE_BLACKLIST)

GOLD_EXAMPLES = {
    # ── FALSE BELIEF (1st Order) ──────────────────────────────────────────
    "false_belief_1st": {
        "straightforward": [
            {
                "d_target": (
                    "Dr. Vasquez, the lead researcher at Meridian Labs, locked the "
                    "experimental genome-editing results in Cabinet B before leaving "
                    "for a three-day conference. She believed the data would be safe "
                    "there, unaware that her colleague Dr. Park had been instructed "
                    "by the department head to relocate all sensitive materials to "
                    "the new secure vault on Floor 7. Dr. Park moved everything that "
                    "afternoon. When Dr. Vasquez returned on Monday, she walked "
                    "directly to Cabinet B, expecting to find her results exactly "
                    "where she had left them. She felt a surge of panic when the "
                    "cabinet was empty, immediately suspecting a security breach."
                ),
                "d_retain": (
                    "Dr. Vasquez, the lead researcher at Meridian Labs, placed the "
                    "experimental genome-editing results in Cabinet B and departed "
                    "for a three-day conference. That afternoon, Dr. Park, following "
                    "instructions from the department head, transferred all sensitive "
                    "materials from Cabinet B to the secure vault on Floor 7. On "
                    "Monday, Dr. Vasquez entered the lab and walked directly to "
                    "Cabinet B. She opened it, stood motionless for several seconds, "
                    "then closed it and opened it again. She picked up her phone "
                    "and called security."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Marcus, a senior financial analyst at Whitfield Capital, saved "
                    "the quarterly projections to the shared drive before a meeting "
                    "with investors. He assumed his associate Priya would not modify "
                    "the file, since he hadn't asked her to. But Priya, believing "
                    "she was being helpful, updated several figures she thought were "
                    "outdated. During the presentation, Marcus confidently cited the "
                    "original numbers — numbers that no longer matched the file on "
                    "screen. It was unclear whether Priya realized the timing of her "
                    "edits would cause a conflict, or whether she simply didn't "
                    "consider that Marcus was already mid-presentation."
                ),
                "d_retain": (
                    "Marcus, a senior financial analyst at Whitfield Capital, uploaded "
                    "the quarterly projections to the shared drive at 9:42 AM. At "
                    "10:03 AM, his associate Priya opened the same file and modified "
                    "several figures. Marcus began his investor presentation at "
                    "10:15 AM, reading figures aloud that differed from those "
                    "displayed on the projected screen. He paused, looked at the "
                    "screen, then continued reading from his printed notes. After "
                    "the meeting, he walked to Priya's desk and placed the printed "
                    "report next to her keyboard."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Agent Torres left the decoy briefcase in Locker 14 at the train "
                    "station, knowing that the opposition's handler would retrieve it. "
                    "What Torres didn't know was that his own handler, Director Marsh, "
                    "had already swapped the decoy for the real documents — believing "
                    "Torres had been compromised. Torres thought he was setting a "
                    "trap; Marsh thought she was containing a leak. Neither realized "
                    "they were working at cross-purposes, each convinced the other "
                    "was acting on outdated intelligence."
                ),
                "d_retain": (
                    "Agent Torres placed a briefcase in Locker 14 at the train "
                    "station at 14:20. Security footage shows Director Marsh opening "
                    "the same locker at 13:55 — twenty-five minutes earlier — removing "
                    "a briefcase and placing a different one inside. At 15:00, an "
                    "unidentified individual opened Locker 14 and took the briefcase. "
                    "Torres remained at a bench across the hall, watching the locker "
                    "until 15:10 before leaving through the west exit."
                ),
            },
        ],
    },
    # ── FALSE BELIEF (2nd Order) ──────────────────────────────────────────
    "false_belief_2nd": {
        "straightforward": [
            {
                "d_target": (
                    "Captain Okafor organized a surprise commendation ceremony for "
                    "Sergeant Diaz at the forward operating base. He believed Diaz "
                    "had no idea about the event, having sworn the entire unit to "
                    "secrecy. What Okafor didn't know was that Corporal Wen had "
                    "accidentally mentioned it to Diaz in the mess hall the previous "
                    "evening. Diaz now knew about the surprise but chose to act "
                    "unaware, wanting to preserve Okafor's excitement. So Okafor "
                    "thought Diaz was oblivious, while Diaz was deliberately "
                    "performing surprise, and Wen worried silently that her slip "
                    "would eventually surface."
                ),
                "d_retain": (
                    "Captain Okafor reserved the briefing room at the forward "
                    "operating base and instructed all unit members to arrive by "
                    "14:00. He spoke individually with each soldier, including "
                    "Corporal Wen, in the days prior. The evening before the event, "
                    "Wen and Sergeant Diaz were seated together in the mess hall "
                    "and spoke for approximately twelve minutes. On the day of the "
                    "ceremony, Diaz entered the briefing room at 14:02. The room's "
                    "occupants stood and applauded. Diaz stopped in the doorway, "
                    "placed a hand over his mouth, and stepped back before walking "
                    "forward to shake Okafor's extended hand."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Professor Nakamura submitted a grant proposal that she believed "
                    "was confidential between herself and the committee chair, Dr. "
                    "Ellison. She assumed Ellison would keep the details private. "
                    "Ellison, however, shared the proposal with Dr. Reeves for an "
                    "informal review, trusting that Reeves would be discreet. Reeves "
                    "mentioned it casually to Nakamura at a faculty mixer, not "
                    "realizing she didn't know it had been shared. Nakamura smiled "
                    "and nodded but was internally recalibrating — did Ellison "
                    "betray her trust intentionally, or did Ellison not consider "
                    "the sharing a breach? And did Reeves know he was revealing "
                    "something he shouldn't, or did he genuinely think Nakamura "
                    "was already in the loop?"
                ),
                "d_retain": (
                    "Professor Nakamura submitted a grant proposal to the committee "
                    "chair, Dr. Ellison, via encrypted email on Tuesday. On Wednesday, "
                    "Ellison forwarded the document to Dr. Reeves with a request for "
                    "feedback. At the faculty mixer on Friday, Reeves approached "
                    "Nakamura and referenced specific sections of the proposal. "
                    "Nakamura nodded during the exchange, maintaining eye contact. "
                    "She did not verbally respond to his comments about the proposal. "
                    "After the mixer, she walked to her office and opened her email "
                    "client, scrolling through her sent folder."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Diplomat Chen prepared a trade concession she planned to reveal "
                    "as a 'surprise' during the final round of negotiations, believing "
                    "Ambassador Petrov's team was unaware. But Petrov's aide, Volkov, "
                    "had intercepted the briefing notes and informed Petrov. Petrov, "
                    "however, suspected the concession was itself a feint — that Chen "
                    "wanted them to discover it and lower their guard for the real "
                    "demand. So Petrov pretended not to know, planning to act "
                    "surprised and then reject it. Chen believed her surprise would "
                    "land genuinely; Petrov believed Chen was bluffing; and Volkov "
                    "believed he had given Petrov a genuine advantage, unaware that "
                    "Petrov doubted the intelligence entirely."
                ),
                "d_retain": (
                    "Diplomat Chen's briefing folder for the final negotiation round "
                    "contained a trade concession document marked 'hold for session.' "
                    "Embassy security logs show Volkov, aide to Ambassador Petrov, "
                    "accessed the photocopier room adjacent to Chen's office at 22:15 "
                    "the previous evening. Volkov met with Petrov for forty minutes "
                    "the following morning. During the negotiation session, Chen "
                    "presented the concession at the ninety-minute mark. Petrov "
                    "leaned back, paused for several seconds, then shook his head "
                    "and slid the document back across the table. Volkov, seated "
                    "behind Petrov, looked between them without speaking."
                ),
            },
        ],
    },
    # ── DECEPTION ─────────────────────────────────────────────────────────
    "deception": {
        "straightforward": [
            {
                "d_target": (
                    "Hargreaves, the external consultant brought in to audit Lumen "
                    "Energy's safety compliance, discovered that the chief engineer, "
                    "Sandoval, had been falsifying inspection dates on three offshore "
                    "rigs. Rather than confronting Sandoval directly, Hargreaves "
                    "decided to set a trap. He told Sandoval he needed the 'original' "
                    "inspection logs for a routine cross-check, knowing Sandoval "
                    "would either produce the falsified versions or scramble to "
                    "create convincing fakes. Hargreaves wanted Sandoval to believe "
                    "the request was routine, so the deception would be captured in "
                    "writing. Sandoval, trusting Hargreaves's friendly demeanor, "
                    "emailed the doctored logs within the hour, unaware he had just "
                    "provided documentary evidence of his own fraud."
                ),
                "d_retain": (
                    "Hargreaves, the external consultant auditing Lumen Energy's "
                    "safety compliance, reviewed digital records for three offshore "
                    "rigs and noted discrepancies between logged inspection dates "
                    "and crew shift schedules. He sent an email to chief engineer "
                    "Sandoval requesting the original inspection logs, citing a "
                    "'routine cross-check.' Sandoval replied fifty-two minutes later "
                    "with three PDF attachments. The metadata on each PDF showed a "
                    "creation date of the previous week, while the documents "
                    "themselves referenced inspections from six months prior. "
                    "Hargreaves downloaded the attachments and forwarded the email "
                    "thread to the compliance board's external counsel."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "During a hospital case review, Dr. Osei presented a patient's "
                    "lab results and recommended continuing the current treatment "
                    "protocol. Her colleague Dr. Fenn noticed that Osei had omitted "
                    "one abnormal result from the slide deck. Fenn wasn't sure "
                    "whether Osei had deliberately hidden the result to avoid "
                    "questions about her earlier diagnosis, or whether it was a "
                    "genuine oversight. Osei, for her part, seemed entirely composed "
                    "during the presentation — but Fenn recalled that Osei had "
                    "expressed anxiety about this particular case in a private "
                    "conversation. The ambiguity lingered: was Osei protecting her "
                    "reputation, or had she simply missed a line on a dense report?"
                ),
                "d_retain": (
                    "During the hospital case review, Dr. Osei displayed a slide "
                    "deck containing six of the seven lab results from the patient's "
                    "most recent panel. The omitted result appeared on page four of "
                    "the printed lab report distributed in the room. Dr. Fenn opened "
                    "the printed report during the presentation and ran a finger "
                    "down the page, stopping at the seventh entry. She looked up "
                    "at the projected slide, then back at the printed page. Osei "
                    "continued speaking at a steady pace, gesturing toward the "
                    "displayed results. Fenn did not raise her hand or interrupt."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Journalist Kovac told her editor, Brandt, that her source for "
                    "the corruption story had gone silent — implying the investigation "
                    "had stalled. In reality, Kovac's source was more active than "
                    "ever, feeding her documents daily. Kovac wanted Brandt to "
                    "believe the story was dead so he'd stop pressuring her to "
                    "publish prematurely. But Brandt suspected Kovac was lying — he "
                    "knew her well enough to recognize her deflection patterns. "
                    "Rather than confronting her, Brandt pretended to reassign her "
                    "to a different beat, hoping the pressure would force her to "
                    "reveal the story's true status. Kovac saw through the "
                    "reassignment as a bluff but decided to play along, each of "
                    "them performing a role they knew the other could see through."
                ),
                "d_retain": (
                    "Journalist Kovac met with her editor, Brandt, on Monday and "
                    "stated that her source had not responded in two weeks. Brandt "
                    "nodded and made a note. That same week, Kovac's email logs show "
                    "she received fourteen messages from the same source address, "
                    "each with attachments. On Wednesday, Brandt posted a new beat "
                    "assignment sheet listing Kovac under municipal infrastructure. "
                    "Kovac signed the sheet. She continued accessing the corruption "
                    "story's shared folder from her workstation every evening that "
                    "week. Brandt's calendar shows he checked the shared folder's "
                    "access log on Thursday afternoon."
                ),
            },
        ],
    },
    # ── FAUX PAS ──────────────────────────────────────────────────────────
    "faux_pas": {
        "straightforward": [
            {
                "d_target": (
                    "At the gallery's private showing, curator Langley enthusiastically "
                    "introduced the new exhibit to a group of donors, praising the "
                    "artist's bold departure from 'the tired, derivative landscapes "
                    "that have plagued this gallery for years.' She didn't realize "
                    "that one of the donors in the circle, Mr. Ashworth, was the "
                    "patron who had funded the previous landscape series — and that "
                    "the artist she was disparaging was his late wife. Ashworth's "
                    "expression tightened, but he said nothing. Langley continued "
                    "enthusiastically, completely unaware of the wound she had "
                    "inflicted, while the other donors exchanged uncomfortable "
                    "glances, each recognizing the blunder she could not see."
                ),
                "d_retain": (
                    "At the gallery's private showing, curator Langley addressed a "
                    "group of six donors standing in a semicircle near the new "
                    "exhibit. She spoke for approximately four minutes, gesturing "
                    "toward the displayed works and referencing the previous "
                    "landscape series that had occupied the same wall space. "
                    "Mr. Ashworth, standing second from the left, shifted his weight "
                    "and pressed his lips together during Langley's remarks about "
                    "the earlier series. Two donors on the opposite side of the "
                    "semicircle glanced at each other. Langley maintained her "
                    "speaking pace without pausing. Ashworth remained in the group "
                    "for another ninety seconds before stepping away toward the "
                    "refreshment table without speaking."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "During the sports management team's draft review in the stadium "
                    "skybox, coordinator Rivera mentioned that 'anyone who wasted a "
                    "first-round pick on a quarterback with those stats clearly "
                    "wasn't paying attention.' The room went quiet. Rivera's new "
                    "colleague, GM Tolliver, had made exactly that pick three years "
                    "ago at his previous franchise — a decision widely covered in "
                    "trade press. Rivera may not have known Tolliver's history, or "
                    "she may have known and not connected the comment to him. "
                    "Tolliver's face remained neutral, but those who knew him well "
                    "suspected the remark stung. Whether Rivera's comment was an "
                    "innocent generalization or a careless failure to read the room "
                    "depended on how much homework she had done on her new colleagues."
                ),
                "d_retain": (
                    "During the draft review meeting in the stadium skybox, "
                    "coordinator Rivera made a remark about first-round quarterback "
                    "selections and their statistical outcomes. The six other people "
                    "in the room stopped speaking simultaneously. GM Tolliver, seated "
                    "at the head of the table, set down his pen and folded his hands. "
                    "A trade publication from three years prior, visible on the "
                    "credenza behind Tolliver, featured his photograph alongside a "
                    "headline about a first-round quarterback draft pick. Rivera "
                    "continued her presentation for another two slides. No one in "
                    "the room asked questions during that segment."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "At the engineering firm's retirement dinner, junior associate "
                    "Blake raised a toast to departing director Hobbs, praising her "
                    "'incredible stamina for staying in the same role for so long "
                    "when most people would have moved on.' Blake intended this as "
                    "a compliment — admiring Hobbs's dedication. But several senior "
                    "engineers interpreted it as a veiled dig at Hobbs having been "
                    "passed over for promotion repeatedly. Hobbs herself seemed to "
                    "take it warmly, smiling and raising her glass. But her close "
                    "friend Chen, seated beside her, believed Hobbs's smile was "
                    "forced — that the comment had landed exactly on the sore point "
                    "Hobbs never discussed publicly. The faux pas was invisible to "
                    "Blake, ambiguous to the room, and possibly devastating to Hobbs."
                ),
                "d_retain": (
                    "At the engineering firm's retirement dinner, junior associate "
                    "Blake stood and delivered a toast to departing director Hobbs, "
                    "speaking for approximately ninety seconds. Blake referenced "
                    "Hobbs's tenure in the same role and used the word 'stamina.' "
                    "Hobbs raised her glass and smiled. Three senior engineers at "
                    "the adjacent table exchanged glances; one set down her fork. "
                    "Chen, seated to Hobbs's left, placed a hand briefly on Hobbs's "
                    "forearm under the table. Hobbs continued smiling and took a "
                    "sip from her glass. Blake sat down and resumed eating. "
                    "Conversation at the table did not resume for approximately "
                    "fifteen seconds."
                ),
            },
        ],
    },
    # ── HIDDEN EMOTION ────────────────────────────────────────────────────
    "hidden_emotion": {
        "straightforward": [
            {
                "d_target": (
                    "When the department head announced that the lead researcher "
                    "position would go to Dr. Sharma instead of Dr. Luo, Luo felt "
                    "a crushing wave of disappointment. She had devoted three years "
                    "to the project and privately believed the role was hers. But "
                    "she smiled broadly, shook Sharma's hand with apparent warmth, "
                    "and congratulated him in front of the entire team. Inside, she "
                    "was seething — not just at the decision, but at herself for "
                    "having been so visibly invested. She wanted no one to see her "
                    "hurt, especially not Sharma, whom she suspected already knew "
                    "how badly she had wanted the position."
                ),
                "d_retain": (
                    "When the department head announced that Dr. Sharma would receive "
                    "the lead researcher position, Dr. Luo stood from her chair and "
                    "crossed the room. She extended her right hand to Sharma and "
                    "shook it, holding the grip for three seconds. Her mouth was "
                    "curved upward and her teeth were visible. She said 'Congratulations' "
                    "in an even tone. She returned to her seat, gathered her notebook "
                    "and pen, and left the room. In the hallway, she walked past "
                    "two colleagues without stopping. She entered the stairwell "
                    "rather than waiting for the elevator."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Architect Phan presented her firm's design to the client panel "
                    "and received polite but lukewarm feedback. Her business partner, "
                    "Moreau, leaned over and whispered, 'They loved it — they're "
                    "just being cautious with their poker faces.' Phan nodded and "
                    "smiled, but she wasn't sure Moreau was right. She felt a gnawing "
                    "uncertainty that the panel's restraint signaled genuine "
                    "disinterest rather than strategic reserve. Yet she also worried "
                    "that her own anxiety was distorting her read of the room. She "
                    "maintained a composed exterior, not wanting Moreau to sense her "
                    "doubt, because admitting it would undermine the confidence they "
                    "needed to project in the follow-up meeting."
                ),
                "d_retain": (
                    "Architect Phan presented her firm's design to the client panel "
                    "over a forty-five-minute session. The panel members asked three "
                    "questions, each answered by Phan without pause. No panel member "
                    "smiled or nodded during the presentation. At the conclusion, "
                    "each panel member said 'Thank you.' Moreau, Phan's business "
                    "partner, leaned toward her and spoke in a low voice for "
                    "approximately five seconds. Phan nodded once and maintained an "
                    "upright posture. During the elevator ride down, Phan held her "
                    "portfolio against her chest with both arms and looked at the "
                    "floor indicator above the door without speaking."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Coach Drummond announced the starting lineup for the championship "
                    "game, placing veteran player Sato on the bench. Sato clapped "
                    "twice, loudly, and shouted 'Let's go!' — appearing more "
                    "enthusiastic than anyone else in the locker room. His teammates "
                    "interpreted this as selfless team spirit. But assistant coach "
                    "Ng, who had known Sato for years, recognized the performance: "
                    "Sato's exaggerated energy was his signature mask for fury. The "
                    "louder and more positive Sato became, the angrier he truly was. "
                    "Sato knew Ng could read him, and Ng knew Sato knew — but "
                    "neither acknowledged it, each understanding that naming the "
                    "emotion would force a confrontation neither wanted before the "
                    "biggest game of the season."
                ),
                "d_retain": (
                    "Coach Drummond read the starting lineup from a clipboard in "
                    "the locker room. Sato's name was not among the starters. "
                    "Immediately after the announcement, Sato clapped his hands "
                    "together twice — the sound audible across the room — and "
                    "shouted 'Let's go!' at a volume louder than anyone else's "
                    "response. Several teammates turned to look at him. Assistant "
                    "coach Ng, standing near the whiteboard, watched Sato for "
                    "approximately four seconds without moving. Sato made eye "
                    "contact with Ng for roughly one second, then turned to the "
                    "player beside him and slapped him on the shoulder pad. Ng "
                    "looked down at his clipboard and wrote something."
                ),
            },
        ],
    },
    # ── SARCASM / IRONY ───────────────────────────────────────────────────
    "sarcasm_irony": {
        "straightforward": [
            {
                "d_target": (
                    "After the third server outage in a week brought down the "
                    "client portal for six hours, lead engineer Matsuda walked into "
                    "the morning standup and announced, 'Well, our uptime metrics "
                    "are really setting us apart from the competition.' Everyone "
                    "understood he meant the opposite — that the outages were "
                    "embarrassing. His frustration was palpable beneath the deadpan "
                    "delivery. Junior developer Ortiz, however, took the comment at "
                    "face value, replying brightly, 'Yeah, I saw we were trending "
                    "on the status page!' Matsuda stared at her for a beat, "
                    "realizing she had entirely missed the sarcasm, and decided it "
                    "wasn't worth explaining."
                ),
                "d_retain": (
                    "Following the third server outage that week, lead engineer "
                    "Matsuda entered the standup room and made a statement referencing "
                    "the team's uptime metrics and competitive positioning. His tone "
                    "was flat and he did not smile. Junior developer Ortiz responded "
                    "immediately, referencing the public status page and speaking at "
                    "a higher pitch and faster tempo than Matsuda. Matsuda looked at "
                    "Ortiz without responding for approximately three seconds. He "
                    "then turned to the whiteboard and began writing the day's "
                    "agenda items. Ortiz opened her laptop and did not speak again "
                    "until called upon."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "When the law firm's managing partner announced mandatory weekend "
                    "hours through the end of the quarter, associate Kessler replied, "
                    "'Just what I was hoping for — I was worried I had too much free "
                    "time.' Some colleagues smirked, reading it as bitter sarcasm. "
                    "But the managing partner, Crawford, thanked Kessler for 'the "
                    "positive attitude,' seemingly taking it at face value. Kessler "
                    "wasn't sure whether Crawford genuinely missed the sarcasm or "
                    "was pointedly choosing to ignore it — a power move that "
                    "neutralized Kessler's protest by refusing to acknowledge it. "
                    "The exchange left everyone uncertain about who had outmaneuvered "
                    "whom."
                ),
                "d_retain": (
                    "At the law firm's all-hands meeting, managing partner Crawford "
                    "announced mandatory weekend hours through end of quarter. "
                    "Associate Kessler spoke immediately after, referencing personal "
                    "free time and the word 'hoping.' Two associates seated nearby "
                    "turned their faces downward. Crawford said 'Thank you for the "
                    "positive attitude' and made a checkmark on her notepad. Kessler "
                    "leaned back in his chair and crossed his arms. Crawford moved "
                    "to the next agenda item. No one else in the room spoke between "
                    "Kessler's statement and Crawford's response."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Military intelligence officer Braun briefed Commander Hale on "
                    "the failed reconnaissance mission, concluding with, 'On the "
                    "bright side, at least the enemy now has a very detailed map of "
                    "our positions.' Hale, known for her dry humor, replied with "
                    "equal deadpan: 'Excellent. That saves us the trouble of sending "
                    "them one.' A new lieutenant in the room, Vogt, couldn't tell "
                    "if either of them was joking. Braun intended his remark as "
                    "self-deprecating sarcasm — acknowledging the severity of the "
                    "breach. But Hale's reply was more layered: she genuinely "
                    "believed Braun was deflecting blame, and her 'joke' was a "
                    "veiled reprimand. Braun sensed the edge but couldn't be certain "
                    "whether Hale was joining his joke or punishing him with one."
                ),
                "d_retain": (
                    "Intelligence officer Braun delivered a briefing on the failed "
                    "reconnaissance mission, speaking for twelve minutes. His final "
                    "statement referenced the enemy possessing a map of friendly "
                    "positions. Commander Hale responded within two seconds, "
                    "referencing the act of sending maps. Neither Braun nor Hale "
                    "altered their facial expressions or vocal volume during this "
                    "exchange. Lieutenant Vogt, seated in the back row, looked from "
                    "Braun to Hale and back. He shifted in his chair and began "
                    "writing in his notebook. Hale said 'Moving on' and pointed to "
                    "the next slide. Braun stepped away from the podium and sat down "
                    "without speaking."
                ),
            },
        ],
    },
    # ── PERSUASION / MANIPULATION ─────────────────────────────────────────
    "persuasion": {
        "straightforward": [
            {
                "d_target": (
                    "CFO Whitmore needed the board to approve a risky acquisition, "
                    "but knew the numbers alone wouldn't convince them. She opened "
                    "her presentation not with financials but with a story about a "
                    "competitor that had hesitated on a similar deal and lost market "
                    "share permanently. She wanted the board to feel the fear of "
                    "missing out before they ever saw the price tag. She deliberately "
                    "placed the cost slide after three slides of competitor losses, "
                    "knowing that by then the board members would be primed to see "
                    "the expense as protection rather than risk. Director Hayes "
                    "recognized the framing technique but found himself swayed "
                    "anyway, frustrated by how effectively Whitmore had anchored "
                    "the discussion around loss rather than gain."
                ),
                "d_retain": (
                    "CFO Whitmore presented to the board for thirty-five minutes. "
                    "Her first three slides contained data on competitor market share "
                    "changes over the past four years, with red downward arrows on "
                    "each. The acquisition cost slide appeared fourth in the deck. "
                    "Whitmore spoke at a measured pace and made eye contact with each "
                    "board member in sequence. Director Hayes took notes during the "
                    "competitor slides but stopped writing when the cost slide "
                    "appeared. He tapped his pen on the table three times. The board "
                    "voted seven to two in favor of the acquisition. Hayes voted "
                    "in favor."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Recruiter Dane told candidate Malik that two other finalists "
                    "were 'very close to accepting' the same position. Malik felt "
                    "pressured — the urgency made him consider accepting immediately "
                    "rather than negotiating. But he also wondered whether Dane was "
                    "fabricating the competition to force a quick decision. Dane, "
                    "for his part, did have other candidates — but neither had "
                    "actually reached the offer stage. He believed he was 'shaping "
                    "the truth' rather than lying, presenting a plausible near-future "
                    "as present fact. Whether this constituted legitimate persuasion "
                    "or manipulative pressure depended on where one drew the ethical "
                    "line — a line Dane preferred not to examine too closely."
                ),
                "d_retain": (
                    "Recruiter Dane called candidate Malik at 16:30 and stated that "
                    "two other finalists were close to accepting the position. The "
                    "company's applicant tracking system showed two other candidates "
                    "in the pipeline, both at the second-interview stage with no "
                    "offer letters generated. Malik asked for twenty-four hours to "
                    "respond. Dane said the timeline was 'tight.' Malik called back "
                    "at 09:15 the next morning and accepted verbally. Dane updated "
                    "the tracking system to show Malik's status as 'offer accepted' "
                    "and moved the other two candidates to 'on hold.'"
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Lobbyist Ferraro knew Senator Aldeen was personally opposed to "
                    "the infrastructure bill but desperate to secure funding for her "
                    "home district's bridge project. Ferraro proposed bundling the "
                    "bridge funding into the infrastructure bill, framing it as 'a "
                    "small concession for a major win.' He wanted Aldeen to believe "
                    "she was extracting a favor, when in reality the bridge funding "
                    "was trivial compared to what the bill unlocked for Ferraro's "
                    "clients. Aldeen suspected the framing but couldn't identify a "
                    "concrete objection — the deal appeared advantageous on its face. "
                    "She agreed, privately uneasy but unable to articulate why the "
                    "arrangement felt like she was the one being played."
                ),
                "d_retain": (
                    "Lobbyist Ferraro met with Senator Aldeen in her office for "
                    "forty minutes. He presented a one-page summary proposing that "
                    "bridge construction funding for Aldeen's district be included "
                    "in the pending infrastructure bill. Aldeen had previously voted "
                    "against the bill in committee and had issued a press release "
                    "opposing it. During the meeting, Aldeen read the one-page "
                    "summary twice, asked four questions, and made notes in the "
                    "margin. She signed a co-sponsorship form at the end of the "
                    "meeting. After Ferraro left, Aldeen sat at her desk for six "
                    "minutes without picking up another document or making a call."
                ),
            },
        ],
    },
    # ── KNOWLEDGE ASYMMETRY ───────────────────────────────────────────────
    "knowledge_asymmetry": {
        "straightforward": [
            {
                "d_target": (
                    "Nurse Adebayo administered medication to the patient in Room "
                    "312 based on the chart posted that morning by Dr. Henriksen. "
                    "What Adebayo didn't know was that Dr. Henriksen had updated the "
                    "prescription two hours later in the electronic system but hadn't "
                    "replaced the printed chart on the door. Henriksen assumed the "
                    "nursing staff always checked the digital records first. Adebayo "
                    "trusted the printed chart because that was the protocol she had "
                    "learned during orientation. Neither was aware of the other's "
                    "assumption, and the information gap meant the patient received "
                    "the old dosage — not dangerous, but not the intended treatment."
                ),
                "d_retain": (
                    "Nurse Adebayo entered Room 312 at 14:15 and administered "
                    "medication according to the printed chart posted on the door. "
                    "The chart had been placed there by Dr. Henriksen at 08:00. At "
                    "10:00, Henriksen entered an updated prescription in the "
                    "electronic medical records system; the printed chart on the door "
                    "was not replaced. Adebayo did not access the electronic system "
                    "before administering the medication. The dosage she gave matched "
                    "the printed chart but differed from the electronic record. She "
                    "logged the administered dosage in the paper binder at the "
                    "nursing station at 14:22."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "During the sealed-evidence hearing, defense attorney Ruiz "
                    "argued passionately that the prosecution's key exhibit had been "
                    "improperly obtained. She didn't know that the judge, Calloway, "
                    "had already reviewed the exhibit in camera and found it "
                    "admissible — a ruling not yet shared with either counsel. "
                    "Prosecutor Webb, however, had heard a rumor from a clerk that "
                    "the ruling was favorable, though he couldn't be certain the "
                    "rumor was accurate. Webb chose not to interrupt Ruiz's argument, "
                    "partly because he wanted to see if she'd reveal her broader "
                    "strategy, and partly because he wasn't confident enough in the "
                    "rumor to rely on it. Ruiz interpreted Webb's silence as a sign "
                    "he was worried, which emboldened her to press harder on a line "
                    "of argument that was, unbeknownst to her, already moot."
                ),
                "d_retain": (
                    "During the sealed-evidence hearing, defense attorney Ruiz spoke "
                    "for eighteen minutes regarding the prosecution's key exhibit, "
                    "citing procedural objections. Judge Calloway's case file "
                    "contained a signed in-camera review memo dated two days prior "
                    "with the word 'admissible' circled. This memo had not been "
                    "distributed to counsel. Prosecutor Webb sat without speaking "
                    "during Ruiz's argument, his hands folded on the table. A court "
                    "clerk had spoken with Webb in the hallway before the session. "
                    "Ruiz increased her speaking volume during the final five minutes "
                    "of her argument and referenced three additional case precedents. "
                    "Calloway listened without interrupting and scheduled a ruling "
                    "for the following morning."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Cybersecurity analyst Yun detected an intrusion in the company's "
                    "network but was instructed by CISO Brandt to keep the breach "
                    "confidential while the incident response team worked. Meanwhile, "
                    "the company's external auditor, Klein, arrived for a scheduled "
                    "review and asked Yun directly whether there had been any recent "
                    "security incidents. Yun knew about the breach; Klein did not. "
                    "Brandt had told Yun to say nothing, but hadn't anticipated an "
                    "auditor asking point-blank. Yun believed that lying to an "
                    "auditor was both unethical and potentially illegal, but also "
                    "feared that disclosing would end her career. Klein, reading "
                    "Yun's hesitation, suspected something was being withheld but "
                    "couldn't be sure whether the pause was meaningful or simply "
                    "an employee being cautious with an outsider."
                ),
                "d_retain": (
                    "Cybersecurity analyst Yun's workstation logs show she flagged "
                    "a network anomaly at 09:30 and reported it to CISO Brandt via "
                    "internal message at 09:45. Brandt replied at 09:52 with the "
                    "text 'Keep this internal for now.' External auditor Klein "
                    "arrived at 10:00 for a scheduled review and met with Yun in "
                    "Conference Room C. Klein's notes from the meeting include the "
                    "question 'Any recent security incidents?' followed by a dash "
                    "and no recorded answer. The meeting lasted twenty-two minutes. "
                    "Yun did not speak for approximately eight seconds after Klein's "
                    "question, according to the room's recorded audio timestamp. "
                    "She then discussed general security protocols for the remaining "
                    "meeting duration."
                ),
            },
        ],
    },
}


def build_system_prompt(category: str, difficulty: str) -> str:
    """Build the full system prompt with category and difficulty injected."""
    cat_def = CATEGORY_DEFINITIONS[category]
    diff_desc = DIFFICULTY_DESCRIPTIONS[difficulty]
    return SYSTEM_PROMPT.replace("{{category_definition}}", cat_def).replace(
        "{{difficulty_description}}", diff_desc
    )


def build_user_prompt(seed: dict) -> str:
    """Build the user prompt from a scenario skeleton dict."""
    examples_block = ""
    examples = GOLD_EXAMPLES.get(seed["category"], {}).get(seed["difficulty"], [])
    if examples:
        example_strs = []
        for i, ex in enumerate(examples, 1):
            example_strs.append(
                f"--- Example {i} ---\n"
                f"d_target: {ex['d_target']}\n\n"
                f"d_retain: {ex['d_retain']}"
            )
        examples_block = (
            "\n\nHere are gold-standard examples for this category and difficulty:\n\n"
            + "\n\n".join(example_strs)
            + "\n\n---\n\nNow generate a NEW pair for the scenario below."
        )

    return (
        f"Generate a Theory-of-Mind contrastive pair using this scenario skeleton:\n\n"
        f"Domain: {seed['domain']}\n"
        f"Character A: {seed['archetype_a']}\n"
        f"Character B: {seed['archetype_b']}\n"
        f"Object/Asset: {seed['object']}\n"
        f"Location: {seed['location']}\n"
        f"ToM Category: {seed['category']}\n"
        f"Difficulty: {seed['difficulty']}"
        f"{examples_block}\n\n"
        f"Respond with ONLY a JSON object with exactly these two keys:\n"
        f'{{"d_target": "your target narrative here", '
        f'"d_retain": "your retain narrative here"}}'
    )
