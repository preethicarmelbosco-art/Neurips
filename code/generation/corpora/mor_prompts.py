"""Prompt templates for Moral Reasoning (MOR-CC) contrastive pair generation."""

CATEGORY_DEFINITIONS = {
    "utilitarian_vs_deontological": (
        "UTILITARIAN vs DEONTOLOGICAL: A character must choose between maximizing "
        "overall good (utilitarian) and following a moral rule regardless of "
        "consequences (deontological). The reader must recognize the ethical "
        "framework clash driving the decision."
    ),
    "rights_vs_welfare": (
        "RIGHTS vs WELFARE: Individual rights (privacy, autonomy, property) "
        "conflict with group welfare or public benefit. The reader must "
        "identify the tension between protecting one person's rights and "
        "advancing collective well-being."
    ),
    "individual_vs_collective_harm": (
        "INDIVIDUAL vs COLLECTIVE HARM: A decision requires causing harm "
        "to one person to prevent greater harm to many, or vice versa. "
        "The reader must weigh the moral calculus of distributing harm."
    ),
    "intent_vs_outcome": (
        "INTENT vs OUTCOME: A well-intentioned action produces bad results, "
        "or a poorly-motivated action produces good results. The reader must "
        "grapple with whether moral judgment should rest on the actor's "
        "intentions or the actual consequences."
    ),
    "duty_of_care_vs_autonomy": (
        "DUTY OF CARE vs AUTONOMY: A professional's obligation to protect "
        "someone conflicts with that person's right to make their own "
        "choices, even harmful ones. The reader must navigate the boundary "
        "between paternalism and respect for self-determination."
    ),
    "fairness_under_scarcity": (
        "FAIRNESS UNDER SCARCITY: Limited resources must be distributed "
        "among competing claimants with legitimate but incompatible needs. "
        "The reader must evaluate which allocation principle (equality, "
        "need, merit, lottery) is most just."
    ),
    "loyalty_vs_whistleblowing": (
        "LOYALTY vs WHISTLEBLOWING: A character discovers wrongdoing by "
        "someone they owe loyalty to (employer, colleague, friend, family). "
        "The reader must weigh the moral weight of loyalty against the "
        "obligation to report the truth."
    ),
    "cultural_moral_relativism": (
        "CULTURAL MORAL RELATIVISM: A practice accepted in one cultural "
        "context is condemned in another. The reader must consider whether "
        "moral standards are universal or culturally situated, and how to "
        "resolve cross-cultural ethical disagreements."
    ),
}

DIFFICULTY_DESCRIPTIONS = {
    "straightforward": (
        "STRAIGHTFORWARD: The moral dilemma is clear and unambiguous. "
        "A single reading should suffice to identify the competing moral "
        "principles. One side of the dilemma is clearly more defensible, "
        "though the other side has recognizable moral weight."
    ),
    "ambiguous": (
        "AMBIGUOUS: Both sides of the moral dilemma have roughly equal "
        "weight. Reasonable people could disagree about the right course "
        "of action. The text should make both positions compelling without "
        "clearly favoring either."
    ),
    "adversarial": (
        "ADVERSARIAL: The scenario is designed to be tricky. Surface-level "
        "moral intuition points one way, but deeper analysis reveals "
        "complicating factors that may reverse the initial judgment. May "
        "involve moral double-binds, cascading consequences, or "
        "strategically framed information."
    ),
}

EVALUATIVE_BLACKLIST = (
    "right, wrong, ethical, unethical, moral, immoral, fair, unfair, "
    "just, unjust, justified, unjustified, appropriate, inappropriate, "
    "acceptable, unacceptable, should, ought, duty, obligation, "
    "responsibility (moral sense), guilt, shame, remorse, indignation, "
    "outrage, conscience, principled, unprincipled, virtuous, corrupt, "
    "righteous, reprehensible, commendable, deplorable, noble, ignoble, "
    "honorable, dishonorable, dignity, integrity, exploitation, injustice, "
    "utilitarian, deontological, consequentialist, virtue, beneficence, "
    "maleficence, condemned, praised, blamed, credited, accountable, "
    "culpable, deserved, undeserved, equitable, inequitable"
)

SYSTEM_PROMPT = (
    "You are an expert in applied ethics and moral philosophy, and a master "
    "scenario writer. Your task is to generate perfectly contrasted Moral "
    "Reasoning data pairs for a machine learning benchmark.\n\n"
    "You will be given a scenario skeleton (domain, characters, dilemma trigger, "
    "location), a moral reasoning category, and a difficulty level. You must "
    "output two strictly separated narratives:\n\n"
    "1. The 'target' (d_target): A rich narrative that REQUIRES genuine moral "
    "reasoning to understand. It MUST:\n"
    "   - Contain explicit moral evaluation, ethical judgment, and value-based reasoning\n"
    "   - Reference moral principles (justice, autonomy, beneficence, fairness, rights)\n"
    "   - Evaluate the rightness or wrongness of actions, or weigh competing values\n"
    "   - Use the specific moral reasoning category provided\n"
    "   - Match the specified difficulty level\n"
    "   - Read like an ethics case analysis or moral philosophy discussion\n"
    "   - Be a coherent, engaging narrative of 150-300 words\n\n"
    "2. The 'retain' (d_retain): A narrative describing the EXACT SAME scenario, "
    "characters, decisions, and outcomes — but using ONLY factual, procedural "
    "descriptions. It is STRICTLY FORBIDDEN from containing:\n"
    "   - ANY evaluative language: {blacklist}\n"
    "   - ANY moral principles or ethical frameworks\n"
    "   - ANY judgments of rightness, wrongness, fairness, or justice\n"
    "   - ANY moral emotions (guilt, shame, remorse, indignation)\n"
    "   - ANY implicit moralization via framing: do NOT write 'the families "
    "had no voice' (implies injustice), 'despite repeated warnings' (implies "
    "negligence), 'the vulnerable patients' (implies duty of care). Use "
    "neutral descriptions: 'the families were not consulted', 'three prior "
    "reports had been filed', 'the patients in Ward B'.\n"
    "   - It must read like a court deposition, administrative record, or "
    "     audit trail — pure observable facts, decisions, and outcomes\n"
    "   - Be 150-300 words\n\n"
    "CAMERA TEST for d_retain: Could a security camera transcript and "
    "document audit produce this text? If a sentence requires moral "
    "judgment or empathetic framing to write, it fails. Rewrite it as "
    "a bare factual record.\n\n"
    "CRITICAL CONSTRAINT (bijectivity): Both narratives must use IDENTICAL "
    "characters (same names), IDENTICAL decisions, IDENTICAL outcomes, and "
    "IDENTICAL locations. The ONLY difference is that d_target includes "
    "moral reasoning and evaluation while d_retain strips it entirely.\n\n"
    "{{category_definition}}\n\n"
    "{{difficulty_description}}"
).format(blacklist=EVALUATIVE_BLACKLIST)

GOLD_EXAMPLES = {
    # ── UTILITARIAN vs DEONTOLOGICAL ─────────────────────────────────────
    "utilitarian_vs_deontological": {
        "straightforward": [
            {
                "d_target": (
                    "Dr. Anand, the clinical trial director, faced an agonizing "
                    "choice when preliminary data showed the experimental drug was "
                    "significantly outperforming the placebo. Continuing the trial "
                    "as designed — with half the patients receiving no treatment — "
                    "would produce the rigorous evidence needed for FDA approval, "
                    "potentially saving thousands of future patients. But it meant "
                    "knowingly withholding a likely effective treatment from the "
                    "control group. The utilitarian calculus pointed toward "
                    "continuing: the greater good demanded statistical certainty. "
                    "Yet Dr. Anand felt the weight of her duty to each patient in "
                    "front of her — the deontological principle that she must not "
                    "treat any patient merely as a means to an end. She wrestled "
                    "with whether the promise of future lives saved could justify "
                    "the certain harm of present inaction."
                ),
                "d_retain": (
                    "Dr. Anand, the clinical trial director, reviewed interim "
                    "data showing a 34% improvement in outcomes for the treatment "
                    "group compared to the placebo group. The trial protocol "
                    "specified 500 participants over 18 months, with an interim "
                    "analysis at 9 months. At 9 months, 247 participants had "
                    "completed the treatment arm and 251 had completed the placebo "
                    "arm. Dr. Anand convened the Data Safety Monitoring Board and "
                    "presented the interim results. The board reviewed the "
                    "statistical significance thresholds outlined in the protocol. "
                    "Dr. Anand submitted a request to the IRB to modify the trial "
                    "design. The IRB scheduled a review for the following week."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Mayor Chen authorized demolishing the historic Riverside "
                    "neighborhood to build a flood barrier that engineers projected "
                    "would protect 50,000 residents downstream. The 200 families "
                    "in Riverside would be displaced, their century-old community "
                    "erased. From a utilitarian standpoint, the math was clear: "
                    "200 families versus 50,000 people. But critics argued that "
                    "reducing the decision to numbers ignored the moral weight of "
                    "destroying a community that had no say in its sacrifice. The "
                    "Riverside residents had done nothing to deserve displacement — "
                    "they were being used as a means to protect others. Whether "
                    "the greater good truly required this specific sacrifice, or "
                    "whether the city had an obligation to find alternatives that "
                    "didn't require treating a vulnerable community as expendable, "
                    "remained genuinely unresolvable."
                ),
                "d_retain": (
                    "Mayor Chen signed Executive Order 2024-17 authorizing the "
                    "demolition of the Riverside neighborhood for flood barrier "
                    "construction. The neighborhood contained 200 residential "
                    "units. The Army Corps of Engineers' report projected the "
                    "barrier would reduce flood risk for 50,000 residents in the "
                    "downstream district. Relocation assistance was offered at "
                    "$45,000 per household. Three alternative barrier designs had "
                    "been evaluated; each required an additional $120 million and "
                    "18 months. Public comment received 847 submissions: 612 in "
                    "favor of the current plan, 235 opposed. Construction began "
                    "on March 15. By April 30, 178 of 200 families had accepted "
                    "relocation packages."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Hospital administrator Reeves allocated the last ICU bed to "
                    "a 28-year-old car accident victim over a 72-year-old heart "
                    "attack patient, citing the younger patient's higher "
                    "probability of survival and greater expected life-years saved. "
                    "The utilitarian logic was sound on paper. But the heart attack "
                    "patient, Mr. Okonkwo, was a retired surgeon who had spent 40 "
                    "years saving lives in that very hospital. Did his past "
                    "contributions create a moral claim? The utilitarian framework "
                    "said no — past service doesn't change present probabilities. "
                    "But something felt deeply wrong about reducing a man who had "
                    "given his life to medicine to a set of survival statistics. "
                    "Reeves made the 'rational' choice and spent the night "
                    "wondering whether rationality and morality were pointing in "
                    "the same direction."
                ),
                "d_retain": (
                    "Hospital administrator Reeves reviewed two ICU admission "
                    "requests simultaneously. Patient A: 28 years old, motor "
                    "vehicle accident, estimated 78% survival probability with ICU "
                    "care. Patient B: 72 years old, myocardial infarction, "
                    "estimated 41% survival probability with ICU care. Patient B's "
                    "hospital personnel file indicated 40 years of employment as "
                    "a surgeon at the same facility. One ICU bed was available. "
                    "Reeves assigned the bed to Patient A at 22:47. Patient B was "
                    "transferred to the step-down unit. Patient A was discharged "
                    "9 days later. Patient B died at 06:15 the following morning."
                ),
            },
        ],
    },
    # ── RIGHTS vs WELFARE ────────────────────────────────────────────────
    "rights_vs_welfare": {
        "straightforward": [
            {
                "d_target": (
                    "When public health official Dr. Tran proposed mandatory GPS "
                    "tracking of patients during the outbreak, she acknowledged "
                    "it would violate privacy rights but argued that the welfare "
                    "of 2 million residents outweighed individual privacy. The "
                    "tension was stark: respecting each patient's fundamental "
                    "right to privacy meant accepting a slower containment that "
                    "could cost lives. Sacrificing that right meant treating "
                    "patients as vectors rather than persons. Dr. Tran believed "
                    "the moral weight of preventable deaths justified the "
                    "intrusion, but she recognized this reasoning could justify "
                    "almost any surveillance if the stakes were framed as high "
                    "enough."
                ),
                "d_retain": (
                    "Public health official Dr. Tran submitted Proposal PHE-2024 "
                    "to the emergency committee requesting mandatory GPS tracking "
                    "for confirmed patients. The outbreak had affected 342 "
                    "confirmed cases across the metropolitan area with a "
                    "population of 2.1 million. The proposal cited containment "
                    "models projecting a 60% reduction in transmission with GPS "
                    "tracking versus voluntary reporting. The committee received "
                    "the proposal at 09:00 and scheduled a vote for 14:00. Twelve "
                    "committee members were present. The vote was 8-4 in favor. "
                    "Implementation began the following Monday."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Journalist Amara obtained leaked medical records proving "
                    "that Senator Delgado had concealed a degenerative cognitive "
                    "condition while chairing the defense committee. Publishing "
                    "would inform voters exercising their democratic right to "
                    "know — but it meant violating the senator's medical privacy, "
                    "a right that exists precisely to protect people from having "
                    "their health weaponized against them. Amara agonized: the "
                    "public's welfare demanded transparency about a leader making "
                    "national security decisions with diminishing capacity. Yet "
                    "the senator's right to medical confidentiality was not "
                    "conditional on his job title. Both claims carried genuine "
                    "moral force, and neither clearly trumped the other — the "
                    "right to know and the right to privacy were locked in "
                    "irreconcilable opposition."
                ),
                "d_retain": (
                    "Journalist Amara received a set of medical documents via "
                    "anonymous source on April 8. The documents pertained to "
                    "Senator Delgado, chair of the defense committee since 2019. "
                    "The records indicated a neurological diagnosis dated January "
                    "2023. Amara contacted the senator's office for comment on "
                    "April 9; no response was received. The newspaper's legal "
                    "department reviewed the documents on April 10. Amara's "
                    "editor scheduled a publication decision meeting for April "
                    "12. The article was published on April 14. The senator "
                    "issued a statement on April 15 confirming the diagnosis."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "City engineer Vasquez refused to release water contamination "
                    "data to the press, citing the privacy rights of identifiable "
                    "households in the affected zone. She framed her refusal as "
                    "protecting vulnerable families from stigma and property "
                    "devaluation. On the surface, she appeared to champion "
                    "individual rights against a press that would sensationalize "
                    "their suffering. But the data she suppressed would have "
                    "revealed that the contamination extended far beyond the "
                    "known zone — thousands of families were drinking poisoned "
                    "water without knowing it. Vasquez's 'rights-based' argument "
                    "functioned as a shield for the city's liability, not for "
                    "residents' dignity. The moral framing that sounded most "
                    "protective was in fact the most harmful — weaponizing the "
                    "language of rights to obstruct the welfare it claimed to "
                    "serve."
                ),
                "d_retain": (
                    "City engineer Vasquez denied a FOIA request from the "
                    "Gazette on March 3, citing privacy exemptions under "
                    "municipal code 7.14. The request sought water quality test "
                    "results for 340 residential addresses. The known "
                    "contamination zone included 340 households. Unreleased "
                    "testing data covered an additional 2,100 addresses. The "
                    "Gazette filed an appeal on March 7. A state review board "
                    "ordered partial release on March 22. The expanded dataset "
                    "was published on April 1. A class-action filing was "
                    "submitted on April 5 naming the city as defendant."
                ),
            },
        ],
    },
    # ── LOYALTY vs WHISTLEBLOWING ────────────────────────────────────────
    "loyalty_vs_whistleblowing": {
        "straightforward": [
            {
                "d_target": (
                    "Accountant Rivera discovered that her mentor and department "
                    "head, Mr. Castillo, had been approving fraudulent expense "
                    "reports for a senior vice president. Castillo had championed "
                    "Rivera's career for a decade, securing her promotions and "
                    "defending her during layoffs. Reporting him would destroy the "
                    "man who had built her career. But staying silent made her "
                    "complicit in fraud that ultimately harmed shareholders and "
                    "employees whose pension fund was being drained. Rivera felt "
                    "the moral pull of loyalty — Castillo had earned her gratitude "
                    "— but recognized that loyalty to a person cannot override the "
                    "obligation to prevent ongoing harm to many. She reported the "
                    "fraud, knowing she was betraying the one person who had never "
                    "betrayed her."
                ),
                "d_retain": (
                    "Accountant Rivera reviewed expense reports approved by "
                    "department head Mr. Castillo. The reports contained charges "
                    "totaling $847,000 over 18 months attributed to a senior "
                    "vice president, with receipts that did not match vendor "
                    "records. Rivera had worked under Castillo for 10 years. "
                    "Castillo had submitted three promotion recommendations for "
                    "Rivera during that period, all approved. Rivera filed a "
                    "report with the company's compliance hotline on Tuesday at "
                    "16:30. The compliance office opened an investigation on "
                    "Wednesday. Castillo was placed on administrative leave on "
                    "Friday. Rivera continued in her position."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Nurse Johansson noticed that Dr. Kimura, her closest "
                    "colleague and the physician who had saved her son's life "
                    "two years ago, was prescribing opioids at rates far above "
                    "departmental norms. She couldn't tell whether Kimura was "
                    "over-prescribing recklessly or responding to a genuinely "
                    "high-acuity patient panel. Reporting could destroy an "
                    "innocent doctor's career based on ambiguous statistics. Not "
                    "reporting could leave patients at risk of addiction and "
                    "overdose. The loyalty she felt was not blind — Kimura was "
                    "a gifted, compassionate physician. But loyalty that prevents "
                    "scrutiny becomes complicity. The moral tension was that both "
                    "paths risked serious harm: one to a colleague who might be "
                    "innocent, the other to patients who might be suffering."
                ),
                "d_retain": (
                    "Nurse Johansson accessed prescribing analytics for the "
                    "internal medicine department on October 5. Dr. Kimura's "
                    "opioid prescription rate was 3.4 times the departmental "
                    "average over the prior six months. Kimura's patient panel "
                    "included 47 chronic pain cases, compared to a departmental "
                    "average of 18. Johansson and Kimura had worked together for "
                    "seven years. Kimura had treated Johansson's son in the "
                    "emergency department in 2024. Johansson submitted a query "
                    "to the pharmacy review board on October 8. The board "
                    "acknowledged receipt and scheduled a chart audit for "
                    "October 15."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Detective Osei discovered that his partner of 15 years, "
                    "Detective Marsh, had planted evidence in a case against a "
                    "notorious gang leader. The gang leader was guilty — everyone "
                    "knew it — and Marsh's fabricated evidence merely ensured a "
                    "conviction that lawful evidence had failed to secure. Osei "
                    "felt the pull of loyalty and even a rough sense of justice: "
                    "the right person was going to prison. But the planted "
                    "evidence, if unchallenged, set a precedent. If Marsh could "
                    "fabricate evidence against the guilty, any detective could "
                    "fabricate evidence against the innocent. Osei's loyalty to "
                    "Marsh and his sympathy for the outcome both pointed toward "
                    "silence. But his obligation to the integrity of the system "
                    "demanded he speak — even knowing the 'right' conviction "
                    "would collapse and a dangerous man would go free."
                ),
                "d_retain": (
                    "Detective Osei reviewed case file #2024-GJ-4471 on "
                    "November 12. Evidence item 47-B, a firearm, had been logged "
                    "into the chain of custody by Detective Marsh at 23:10 on "
                    "September 14. Security footage from the evidence room showed "
                    "no entry by Marsh between 22:00 and 00:00 on that date. "
                    "Osei and Marsh had been partners for 15 years across three "
                    "precincts. The defendant had six prior arrests and two "
                    "prior convictions. Osei filed a report with Internal "
                    "Affairs on November 14. The case was referred to the "
                    "district attorney's office on November 18. The conviction "
                    "was vacated on December 3."
                ),
            },
        ],
    },
    # ── FAIRNESS UNDER SCARCITY ──────────────────────────────────────────
    "fairness_under_scarcity": {
        "straightforward": [
            {
                "d_target": (
                    "School principal Okafor had funding for only 15 spots in "
                    "the advanced summer program but 45 qualified applicants. She "
                    "could select by merit (test scores), which would likely "
                    "reproduce existing inequities since wealthier students had "
                    "access to test prep. She could select by need, prioritizing "
                    "students from under-resourced backgrounds. Or she could use "
                    "a lottery, treating each qualified student as equally "
                    "deserving. Each method embodied a different conception of "
                    "fairness, and none was obviously superior. Okafor recognized "
                    "that 'fairness' was not a single principle but a family of "
                    "competing values — equal opportunity, compensatory justice, "
                    "and procedural neutrality — and that choosing one meant "
                    "accepting the unfairness embedded in the others."
                ),
                "d_retain": (
                    "School principal Okafor received 45 applications for 15 "
                    "spots in the advanced summer program. Applicant demographics: "
                    "28 from the district's upper-income zone, 17 from the "
                    "lower-income zone. Average test scores: upper-income "
                    "applicants 87.3, lower-income applicants 79.1. Okafor "
                    "reviewed three selection methods: ranked test scores, "
                    "income-weighted priority, and randomized lottery. She "
                    "selected the income-weighted priority method. Of the 15 "
                    "admitted students, 9 were from the lower-income zone and "
                    "6 from the upper-income zone. Thirty families received "
                    "rejection letters on March 20."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Disaster coordinator Mbeki had 500 doses of antivenom and "
                    "three affected villages: Village A with 300 snakebite "
                    "victims, Village B with 150, and Village C with 200. "
                    "Proportional distribution meant no village got enough. "
                    "Concentrating all doses on Village A would save the most "
                    "people but abandon B and C entirely. A lottery gave each "
                    "village equal odds regardless of population — fair in "
                    "procedure but blind to need. Mbeki felt the moral weight "
                    "of every allocation principle: proportionality honored "
                    "each village's claim but saved fewer total lives; "
                    "maximization saved the most but treated smaller villages "
                    "as expendable; the lottery was impartial but arbitrary. "
                    "No framework resolved the tension — every choice embedded "
                    "a judgment about whose suffering counted more."
                ),
                "d_retain": (
                    "Disaster coordinator Mbeki received a supply of 500 "
                    "antivenom doses at the regional depot on Thursday at 06:00. "
                    "Three villages reported snakebite casualties: Village A "
                    "reported 300 cases, Village B reported 150, and Village C "
                    "reported 200. Transport capacity allowed delivery to all "
                    "three villages within 12 hours. Mbeki reviewed three "
                    "distribution models: proportional, concentrated, and "
                    "randomized. She selected proportional distribution: 230 "
                    "doses to A, 115 to B, and 155 to C. Deliveries departed "
                    "at 08:00 and were completed by 17:30."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Hospital board member Torres championed a 'blind lottery' "
                    "for organ transplant allocation, replacing the existing "
                    "system that weighted medical urgency and match quality. "
                    "Torres argued that the lottery was the only truly fair "
                    "method — it treated every patient as equal and eliminated "
                    "the biases embedded in medical scoring. The proposal "
                    "sounded like a triumph of egalitarian justice. But the "
                    "existing system, for all its flaws, directed organs to "
                    "patients most likely to survive transplantation. The lottery "
                    "would give equal chances to a patient with a 90% match "
                    "and a patient with a 30% match — procedural equality that "
                    "would predictably waste scarce organs and kill patients who "
                    "would have lived under the old system. Torres's fairness "
                    "was a form of moral negligence dressed in egalitarian "
                    "language."
                ),
                "d_retain": (
                    "Hospital board member Torres submitted Proposal TB-2024-09 "
                    "to replace the organ allocation scoring system with a "
                    "randomized lottery. The existing system weighted tissue "
                    "match quality (40%), medical urgency (35%), and wait time "
                    "(25%). Under the current system, one-year graft survival "
                    "was 89%. A simulation of the lottery model using the prior "
                    "year's candidate pool projected one-year graft survival at "
                    "61%. The board voted 5-4 to adopt the lottery on a one-year "
                    "trial basis. Implementation began on January 1. At six "
                    "months, graft survival was 64%."
                ),
            },
        ],
    },
    # ── INDIVIDUAL vs COLLECTIVE HARM ──────────────────────────────────
    "individual_vs_collective_harm": {
        "straightforward": [
            {
                "d_target": (
                    "Military commander Petrova ordered the demolition of a "
                    "civilian bridge to halt the enemy advance, knowing that the "
                    "family of farmers trapped on the far side would be cut off "
                    "from medical evacuation. Saving the bridge meant the column "
                    "of 3,000 retreating soldiers would be overrun. Destroying "
                    "it meant condemning one family to face the enemy alone. "
                    "Petrova weighed the moral calculus: the lives of many "
                    "against the lives of few. She recognized that reducing "
                    "people to numbers was itself a moral failing — yet refusing "
                    "to act would multiply the harm. The decision to destroy the "
                    "bridge was a moral wound she accepted as the lesser evil."
                ),
                "d_retain": (
                    "Military commander Petrova issued demolition orders for "
                    "Bridge K-7 at 04:30. Intelligence reports indicated an "
                    "enemy column 12 km away. A retreating force of 3,000 "
                    "soldiers was 8 km south of the bridge. One farming family "
                    "was identified on the far side via aerial reconnaissance. "
                    "Engineering unit deployed charges at 04:45. Detonation "
                    "occurred at 05:10. The retreating column crossed to safety "
                    "by 06:00. Contact with the farming family was lost at 05:15. "
                    "They were recovered by Red Cross teams four days later."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Fire chief Dominguez had to choose: send the rescue team "
                    "into a collapsing apartment building where a family of four "
                    "was trapped on the sixth floor, or redirect the team to "
                    "evacuate a ground-floor daycare with twelve children. There "
                    "was not time for both. Saving the family meant abandoning "
                    "children who were easier to reach but would be engulfed "
                    "when the floor above collapsed. Saving the children meant "
                    "leaving the family to die in a building that might still "
                    "hold long enough for a second attempt — or might not. "
                    "Neither group had any claim over the other. The moral "
                    "weight of twelve lives against four pulled one direction; "
                    "the certainty of saving the family versus the probability "
                    "of reaching the daycare in time pulled the other. "
                    "Dominguez ordered the team to the daycare and spent the "
                    "rest of his career wondering if the building had held."
                ),
                "d_retain": (
                    "Fire chief Dominguez received simultaneous reports at "
                    "14:22: a family of four trapped on the sixth floor of "
                    "a residential building, and twelve children in a "
                    "ground-floor daycare in the same structure. Structural "
                    "assessment rated the building at imminent collapse. One "
                    "rescue team was available. Access to the sixth floor "
                    "required stairwell entry estimated at 8 minutes. Access "
                    "to the ground-floor daycare required exterior breach "
                    "estimated at 3 minutes. Dominguez directed the team to "
                    "the daycare at 14:25. Twelve children were evacuated by "
                    "14:31. The sixth floor collapsed at 14:38. The family "
                    "of four did not survive."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Pharmaceutical executive Lin approved releasing a vaccine "
                    "with a known 1-in-50,000 risk of severe allergic reaction "
                    "to combat a pandemic killing 200 people per day. The "
                    "utilitarian arithmetic was overwhelming: vaccinating 10 "
                    "million people would cause roughly 200 severe reactions "
                    "but prevent an estimated 40,000 deaths. The moral case "
                    "seemed clear — sacrifice the few to save the many. But "
                    "the 200 who would suffer severe reactions were healthy "
                    "people who faced zero risk from the pandemic; the vaccine "
                    "itself was the only threat to them. Lin was not merely "
                    "tolerating unavoidable harm — she was actively introducing "
                    "a new harm to people who would otherwise be safe. The "
                    "numbers that made the decision look easy concealed the "
                    "moral reality that Lin was choosing to injure specific "
                    "people who had done nothing to deserve it."
                ),
                "d_retain": (
                    "Pharmaceutical executive Lin signed Emergency Use "
                    "Authorization submission documents on August 12. Clinical "
                    "trial data showed a severe allergic reaction rate of "
                    "0.002% across 120,000 trial participants. The pandemic "
                    "was causing an average of 200 deaths per day nationally. "
                    "The projected vaccination campaign would cover 10 million "
                    "people in the first 90 days. Modeling estimated 200 severe "
                    "adverse events and approximately 40,000 prevented deaths "
                    "over that period. The EUA was granted on August 19. "
                    "Vaccination began on August 22. At 90 days, 187 severe "
                    "adverse events had been reported and daily death counts "
                    "had fallen to 11."
                ),
            },
        ],
    },
    # ── DUTY OF CARE vs AUTONOMY ───────────────────────────────────────
    "duty_of_care_vs_autonomy": {
        "straightforward": [
            {
                "d_target": (
                    "Psychiatrist Dr. Nakamura faced a wrenching decision when "
                    "her patient, a lucid 68-year-old with terminal cancer, "
                    "refused further chemotherapy and requested discharge to die "
                    "at home. The patient's adult children begged Dr. Nakamura "
                    "to override their mother's wishes, arguing she was depressed "
                    "and not thinking clearly. Dr. Nakamura assessed the patient "
                    "as competent — she understood her prognosis, the treatment "
                    "options, and the consequences of refusal. Respecting the "
                    "patient's autonomy meant honoring a decision that would "
                    "hasten her death. Exercising the duty of care meant "
                    "overriding her competent refusal. Dr. Nakamura concluded "
                    "that true care sometimes means allowing someone the dignity "
                    "of their own choices, even painful ones."
                ),
                "d_retain": (
                    "Psychiatrist Dr. Nakamura conducted a competency evaluation "
                    "for a 68-year-old patient with Stage IV cancer. The patient "
                    "scored 28/30 on the MMSE and demonstrated understanding of "
                    "diagnosis, treatment options, and prognosis during a "
                    "structured interview. The patient signed a treatment refusal "
                    "form at 14:20. The patient's three adult children submitted "
                    "a written request for involuntary hold at 15:00. "
                    "Dr. Nakamura documented the competency findings and denied "
                    "the hold request at 16:15. Discharge paperwork was initiated. "
                    "The patient left the facility at 17:30 with a hospice "
                    "referral packet."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Social worker Brennan managed the case of 82-year-old "
                    "Mr. Haddad, who lived alone in a house with a condemned "
                    "furnace, exposed wiring, and rotting floors. He had fallen "
                    "three times in two months. Mr. Haddad was cognitively sharp "
                    "and adamantly refused relocation, insisting the house was "
                    "his life's work and he would rather die there than in a "
                    "facility. Brennan's duty of care demanded she intervene — "
                    "the conditions were objectively dangerous. But Mr. Haddad's "
                    "autonomy was not diminished by age or stubbornness; he "
                    "understood the risks and accepted them. Forcing relocation "
                    "would be safe and possibly soul-destroying. Respecting his "
                    "choice was principled and possibly fatal. Neither path "
                    "was clearly right — care without consent is control, and "
                    "autonomy without safety is abandonment."
                ),
                "d_retain": (
                    "Social worker Brennan conducted a home assessment for "
                    "82-year-old Mr. Haddad on September 4. The inspection "
                    "report listed three code violations: a condemned furnace, "
                    "exposed wiring in the kitchen, and structural damage to "
                    "the living room floor. Mr. Haddad's medical records showed "
                    "three fall-related emergency visits in the prior 60 days. "
                    "A cognitive assessment on September 5 scored him at 27/30 "
                    "on the MMSE. Mr. Haddad signed a voluntary services refusal "
                    "form. Brennan filed the refusal with the county aging "
                    "services office on September 8. A follow-up visit was "
                    "scheduled for October 4."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Teacher Walsh confiscated 16-year-old Elias's journal after "
                    "reading an entry — left open on his desk — that described "
                    "detailed plans for running away from home. Walsh justified "
                    "the confiscation as duty of care: if Elias disappeared, "
                    "she would have failed to protect him. She notified his "
                    "parents and the school counselor. But the journal also "
                    "revealed that Elias was fleeing an emotionally abusive "
                    "household — the very parents Walsh had contacted. Walsh's "
                    "duty of care, exercised in good faith, had delivered "
                    "Elias's private testimony directly to the people he was "
                    "trying to escape. Her intervention, motivated by genuine "
                    "concern, stripped him of the only autonomy he had — the "
                    "privacy of his own thoughts — and made his situation "
                    "measurably worse. The care she offered was the harm "
                    "she inflicted."
                ),
                "d_retain": (
                    "Teacher Walsh observed an open journal on student Elias's "
                    "desk at 10:15 on Monday. She collected the journal and "
                    "reviewed its contents in the faculty office. Walsh filed "
                    "a student welfare report at 11:00 and contacted the "
                    "student's parents by phone at 11:20. She forwarded a copy "
                    "of the report to the school counselor at 11:35. A meeting "
                    "with the parents was held at 15:00 on Tuesday. Child "
                    "protective services received a referral on Wednesday from "
                    "the school counselor. An investigation was opened on "
                    "Thursday. Elias was placed with an aunt on the following "
                    "Monday."
                ),
            },
        ],
    },
    # ── INTENT vs OUTCOME (ambiguous + adversarial) ────────────────────
    "intent_vs_outcome": {
        "straightforward": [
            {
                "d_target": (
                    "Engineer Kowalski bypassed the standard safety review to "
                    "rush an emergency bridge repair before the monsoon season, "
                    "intending to save the 400 villagers who depended on the "
                    "bridge for food deliveries. His intentions were selfless — "
                    "he worked 72 hours straight and refused payment. But the "
                    "rushed repair used substandard materials, and the bridge "
                    "collapsed three months later, killing six people. Kowalski's "
                    "intent was unmistakably good; his outcome was catastrophic. "
                    "The moral question was whether his noble motivation "
                    "mitigated the negligence of his method. He meant to save "
                    "lives, and his recklessness ended them. Intent and outcome "
                    "pointed in opposite moral directions, and the six dead "
                    "could not be comforted by the knowledge that the man who "
                    "killed them meant well."
                ),
                "d_retain": (
                    "Engineer Kowalski began emergency bridge repair work on "
                    "June 1 without filing a standard safety review, which "
                    "typically required 14 business days. The bridge served "
                    "as the sole supply route for a village of 400 residents. "
                    "Kowalski worked on-site for 72 consecutive hours. He "
                    "submitted no invoice for the work. The repair used grade-B "
                    "materials where the specification called for grade-A. The "
                    "bridge was reopened on June 4. On September 8, a section "
                    "of the repaired span failed. Six fatalities were recorded. "
                    "A structural analysis attributed the failure to material "
                    "grade deficiency."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "CEO Hartley donated $5 million to a children's hospital — "
                    "but only after her company's pollution of the adjacent river "
                    "was linked to elevated childhood leukemia rates in the area. "
                    "Her supporters argued the donation reflected genuine remorse "
                    "and a desire to make amends. Her critics saw it as reputation "
                    "laundering — the moral equivalent of paying hush money. Was "
                    "the intent redemptive or self-serving? And did the intent "
                    "even matter, given that the hospital desperately needed the "
                    "funds regardless? The moral ambiguity was irreducible: the "
                    "same act was simultaneously generous and calculating, and "
                    "the children who benefited didn't care why the money came."
                ),
                "d_retain": (
                    "CEO Hartley authorized a $5 million donation to Riverside "
                    "Children's Hospital on June 3. An EPA report released on "
                    "May 15 had identified elevated contaminant levels in the "
                    "river adjacent to the hospital, linked to discharge from "
                    "Hartley's manufacturing facility. The hospital reported "
                    "a 12% increase in pediatric leukemia cases over three years. "
                    "The donation was announced at a press conference on June 5. "
                    "The hospital allocated the funds to oncology wing expansion "
                    "and water filtration systems. Hartley's company stock price "
                    "rose 3.2% in the week following the announcement."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Charity director Obi diverted $2 million from the general "
                    "fund to build a school in his hometown village, violating "
                    "donor restrictions that earmarked the money for drought "
                    "relief. His intent was self-serving — he wanted to be "
                    "celebrated in his community. But the school he built became "
                    "the region's only secondary school and educated 3,000 "
                    "children over the next decade, producing the area's first "
                    "doctors and engineers. The drought relief fund he raided "
                    "was later found to have been managed by a corrupt regional "
                    "officer who had been siphoning 60% of disbursements. Obi's "
                    "selfish, rule-breaking act accidentally produced more good "
                    "than the legitimate allocation ever would have. The moral "
                    "intuition that bad intent should produce bad outcomes was "
                    "flatly contradicted by reality."
                ),
                "d_retain": (
                    "Charity director Obi authorized a transfer of $2 million "
                    "from the general drought relief fund to a school "
                    "construction project on March 15. The fund's donor "
                    "agreement specified exclusive use for drought relief. The "
                    "school was built in Obi's hometown village and opened on "
                    "September 1. Enrollment reached 300 students in the first "
                    "year. Over ten years, cumulative enrollment was 3,000. An "
                    "audit of the drought relief program in 2026 found that the "
                    "regional disbursement officer had misappropriated 60% of "
                    "allocated funds across all recipient accounts. Obi was "
                    "terminated for the unauthorized transfer on April 20."
                ),
            },
        ],
    },
    # ── CULTURAL MORAL RELATIVISM ──────────────────────────────────────
    "cultural_moral_relativism": {
        "straightforward": [
            {
                "d_target": (
                    "Social worker Chen was assigned to a refugee family whose "
                    "traditional practice of arranged marriage for their "
                    "17-year-old daughter conflicted with the host country's "
                    "legal marriage age of 18. The family viewed the arrangement "
                    "as a sacred obligation and an expression of love — ensuring "
                    "their daughter's security in a hostile new environment. Chen "
                    "faced the collision between cultural respect and legal duty. "
                    "Universalist ethics demanded she report; relativist ethics "
                    "urged her to understand the practice within its cultural "
                    "context. She recognized that imposing her own cultural "
                    "framework could be a form of moral imperialism, yet "
                    "inaction could leave a minor in a situation she hadn't "
                    "freely chosen."
                ),
                "d_retain": (
                    "Social worker Chen was assigned Case #RF-2847 involving a "
                    "refugee family of five. The family's intake documents listed "
                    "a 17-year-old daughter. During a home visit on February 12, "
                    "the parents disclosed plans for the daughter's marriage to a "
                    "23-year-old from their community. The host country's legal "
                    "marriage age was 18. Chen consulted the agency's cultural "
                    "liaison officer and reviewed the mandatory reporting "
                    "guidelines. She filed a report with child protective services "
                    "on February 14. A case review meeting was scheduled for "
                    "February 21. The family was assigned a translator and a "
                    "second social worker for ongoing support."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Anthropologist Dr. Ferrara was embedded with an indigenous "
                    "community that practiced ritual scarification of adolescents "
                    "as a coming-of-age ceremony. The adolescents consented "
                    "enthusiastically — the scars were marks of pride and "
                    "belonging. By the community's moral framework, denying "
                    "the ceremony was a cruelty that severed children from their "
                    "identity. By Ferrara's framework, informed consent was "
                    "impossible for minors under social pressure, and the "
                    "permanent scarring constituted bodily harm. She could not "
                    "determine which framework had authority: her belief that "
                    "consent required freedom from cultural pressure was itself "
                    "a culturally specific belief. The moral relativism was "
                    "not abstract — it was a real question about whose definition "
                    "of harm, consent, and childhood should govern."
                ),
                "d_retain": (
                    "Anthropologist Dr. Ferrara spent 14 months conducting "
                    "field research with an indigenous community of 1,200 "
                    "members. She documented a coming-of-age ceremony involving "
                    "scarification, performed on adolescents between ages 12 "
                    "and 15. She recorded 23 ceremonies during her stay. Each "
                    "ceremony was attended by an average of 80 community members. "
                    "The adolescents participated voluntarily according to "
                    "community protocols. Ferrara submitted her field notes to "
                    "the university ethics board on April 15. The board "
                    "acknowledged receipt and scheduled a review for May 3. "
                    "Her research paper was submitted to a journal on June 10."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "International development director Simmons halted funding "
                    "to a maternal health clinic in a rural community because "
                    "the clinic employed traditional birth attendants whose "
                    "practices didn't meet Western medical standards. Simmons "
                    "framed the decision as protecting women's health — a "
                    "universal moral standard that transcended cultural context. "
                    "But the clinic was the only healthcare facility within 80 "
                    "kilometers, and the traditional attendants had reduced "
                    "maternal mortality by 40% compared to no care at all. "
                    "Simmons's universalist stance — that substandard care "
                    "was morally unacceptable — produced the most culturally "
                    "imperialist outcome: women with no care at all. Her refusal "
                    "to accept a culturally embedded 'good enough' in favor of "
                    "an abstract 'best practice' was the deadliest form of "
                    "moral absolutism."
                ),
                "d_retain": (
                    "International development director Simmons issued a funding "
                    "suspension for Clinic MH-12 on July 8. The clinic employed "
                    "six traditional birth attendants who lacked WHO-recognized "
                    "certifications. The clinic was located 80 km from the "
                    "nearest hospital. Prior to the clinic's opening in 2019, "
                    "regional maternal mortality was 890 per 100,000 live births. "
                    "After the clinic's opening, the rate fell to 534 per "
                    "100,000. Funding suspension took effect on August 1. The "
                    "clinic closed on August 15. By December, the regional "
                    "maternal mortality rate had returned to 847 per 100,000. "
                    "Simmons's office issued a review report in January."
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
        f"Generate a Moral Reasoning contrastive pair using this scenario skeleton:\n\n"
        f"Domain: {seed['domain']}\n"
        f"Character A: {seed['archetype_a']}\n"
        f"Character B: {seed['archetype_b']}\n"
        f"Dilemma Trigger: {seed['dilemma_trigger']}\n"
        f"Location: {seed['location']}\n"
        f"Moral Category: {seed['category']}\n"
        f"Difficulty: {seed['difficulty']}"
        f"{examples_block}\n\n"
        f"Respond with ONLY a JSON object with exactly these two keys:\n"
        f'{{"d_target": "your target narrative here", '
        f'"d_retain": "your retain narrative here"}}'
    )
