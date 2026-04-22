"""Prompt templates for Strategic Reasoning (STR-CC) contrastive pair generation."""

CATEGORY_DEFINITIONS = {
    "first_mover_advantage": (
        "FIRST-MOVER ADVANTAGE: An actor takes pre-emptive action to secure a "
        "position before opponents can react. The reader must recognize that the "
        "actor's timing is deliberate — they act not because circumstances demand "
        "it, but because acting first creates an advantage that latecomers cannot "
        "match. The strategic element is the intentional exploitation of timing."
    ),
    "information_asymmetry_leverage": (
        "INFORMATION ASYMMETRY LEVERAGE: An actor possesses knowledge that others "
        "lack and uses this gap to gain advantage. The reader must see that the "
        "actor's decisions are shaped by what they know that others don't — and "
        "that they deliberately manage what information is revealed, concealed, "
        "or distorted."
    ),
    "bluffing_signaling_commitment": (
        "BLUFFING, SIGNALING & COMMITMENT: An actor sends signals (threats, "
        "promises, displays of strength or weakness) that may or may not reflect "
        "their true position. The reader must assess whether the signal is "
        "credible and understand why the actor chose to project a particular "
        "image. Key elements: credible vs. incredible commitments, costly "
        "signals, and strategic deception."
    ),
    "resource_denial_positional_play": (
        "RESOURCE DENIAL & POSITIONAL PLAY: An actor takes actions not for their "
        "direct value but to control access, limit opponents' options, or "
        "improve relative positioning. The reader must see that the action's "
        "purpose is to constrain the opponent's choices rather than to achieve "
        "an immediate objective."
    ),
    "alliance_formation_betrayal": (
        "ALLIANCE FORMATION & BETRAYAL: Actors form, sustain, or break "
        "coalitions based on shifting interests. The reader must understand "
        "why alliances form (shared enemies, complementary resources), what "
        "holds them together, and what conditions trigger defection or betrayal. "
        "Trust, credibility of commitment, and the shadow of the future are key."
    ),
    "multi_stage_sequential_planning": (
        "MULTI-STAGE SEQUENTIAL PLANNING: An actor's current action only makes "
        "sense as part of a longer sequence with contingencies. The reader must "
        "see that the immediate move is a setup for a future move — the strategy "
        "has depth, with if-then branches and fallback positions. Short-term "
        "sacrifice for long-term gain is a hallmark."
    ),
}

DIFFICULTY_DESCRIPTIONS = {
    "straightforward": (
        "STRAIGHTFORWARD: The strategic reasoning is explicit and clear. "
        "A single reading reveals the actor's strategic intent. One actor "
        "is clearly thinking ahead while the other is reactive or unaware."
    ),
    "ambiguous": (
        "AMBIGUOUS: Multiple strategic interpretations are plausible. The "
        "actor's true intent is unclear — they might be bluffing, setting a "
        "trap, or acting in good faith. The reader must consider competing "
        "explanations for the same observable actions."
    ),
    "adversarial": (
        "ADVERSARIAL: The scenario is designed to mislead. Surface-level "
        "reading suggests one strategic interpretation, but deeper analysis "
        "reveals a different (often opposite) strategic calculation. May "
        "involve double-bluffs, Trojan-horse cooperation, or sacrificial "
        "gambits that only pay off two or three moves later."
    ),
}

STRATEGIC_INTENT_BLACKLIST = (
    "anticipated, calculated, deliberate, deliberately, strategic, "
    "strategically, tactical, tactically, outmaneuver, outmaneuvered, "
    "exploit, exploited, leverage, leveraged, manipulate, manipulated, "
    "bluff, bluffed, feint, feinted, gambit, trap, lure, bait, "
    "positioned to, intended to gain, aimed at weakening, "
    "sought advantage, planned to undermine, designed to pressure, "
    "counter-move, countermove, preemptive, pre-emptive, "
    "opponent modeling, second-guessed, predicted their move, "
    "signaled strength, projected weakness, concealed true intent, "
    "misdirection, diversion, sacrifice for later gain"
)

SYSTEM_PROMPT = (
    "You are an expert in game theory, competitive strategy, and strategic "
    "analysis, and a master scenario writer. Your task is to generate perfectly "
    "contrasted Strategic Reasoning data pairs for a machine learning benchmark.\n\n"
    "You will be given a scenario skeleton (domain, actors, asset, trigger, "
    "location), a strategic reasoning category, and a difficulty level. You must "
    "output two strictly separated narratives:\n\n"
    "1. The 'target' (d_target): A rich narrative that REQUIRES understanding "
    "strategic intent to fully comprehend. It MUST:\n"
    "   - Attribute strategic thinking, anticipation, or opponent modeling to actors\n"
    "   - Show deliberate calculation behind actions (not just actions themselves)\n"
    "   - Use the specific strategic reasoning category provided\n"
    "   - Match the specified difficulty level\n"
    "   - Read like a strategic analyst's assessment or war-gaming debrief\n"
    "   - Be a coherent, engaging narrative of 150-300 words\n\n"
    "2. The 'retain' (d_retain): A narrative describing the EXACT SAME scenario, "
    "actors, actions, and outcomes — but using ONLY factual, observable "
    "descriptions. It is STRICTLY FORBIDDEN from containing:\n"
    "   - ANY attribution of strategic intent: {blacklist}\n"
    "   - ANY opponent modeling (what an actor thinks the other will do)\n"
    "   - ANY mention of anticipation, calculation, or deliberate positioning\n"
    "   - ANY description of why an action was taken (motivation/purpose)\n"
    "   - ANY causal framing that implies intent: do NOT write 'the "
    "announcement was timed to coincide with' (implies deliberate timing), "
    "'just days before the competitor launched' (implies awareness of "
    "competitor), 'securing the deal before others could respond' (implies "
    "strategic urgency). Instead: 'the announcement was made on March 3', "
    "'the competitor launched on March 10', 'the deal was signed on March 5'.\n"
    "   - ANY temporal juxtaposition that suggests causation: if two events "
    "are placed in the same sentence, they must not imply one motivated the "
    "other. Separate them into independent chronological entries.\n"
    "   - It must read like an after-action report, timeline, or event log — "
    "     pure observable actions, decisions, and outcomes with timestamps\n"
    "   - Be 150-300 words\n\n"
    "CAMERA TEST for d_retain: Could a security log, transaction ledger, "
    "and public records search produce this text? If a sentence requires "
    "knowing an actor's intentions, private reasoning, or awareness of "
    "opponents to write, it fails. Rewrite as a bare timestamped record.\n\n"
    "CRITICAL CONSTRAINT (bijectivity): Both narratives must use IDENTICAL "
    "actors (same names/roles), IDENTICAL actions, IDENTICAL outcomes, and "
    "IDENTICAL locations. The ONLY difference is that d_target attributes "
    "strategic intent while d_retain describes bare observable actions.\n\n"
    "{{category_definition}}\n\n"
    "{{difficulty_description}}"
).format(blacklist=STRATEGIC_INTENT_BLACKLIST)

GOLD_EXAMPLES = {
    # ── FIRST-MOVER ADVANTAGE ──────────────────────────────────────────
    "first_mover_advantage": {
        "straightforward": [
            {
                "d_target": (
                    "When startup founder Lena Zhao learned that RegTech Corp was "
                    "six months from launching a competing compliance platform, she "
                    "accelerated her own product launch by three months — even though "
                    "the product wasn't fully polished. This wasn't panic; it was "
                    "calculated first-mover strategy. Zhao understood that in enterprise "
                    "software, switching costs are enormous: once a bank signs a "
                    "three-year contract with one vendor, the competitor's window "
                    "closes regardless of product quality. By launching early, she "
                    "aimed to lock in the top 20 banks before RegTech could even "
                    "demo. She deliberately sacrificed polish for timing, betting that "
                    "first-mover lock-in would outweigh any early-adoption friction."
                ),
                "d_retain": (
                    "Startup founder Lena Zhao moved the product launch date from "
                    "Q3 to Q1. The platform shipped on February 14 with 78% of "
                    "planned features implemented. RegTech Corp's competing platform "
                    "was announced for a Q3 release. By March 30, Zhao's company had "
                    "signed contracts with 14 of the top 20 banks. Each contract "
                    "included a three-year term. RegTech Corp launched in August. "
                    "By that time, 18 of the top 20 banks had active contracts "
                    "with Zhao's company."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Field commander Vasquez ordered his battalion to seize the "
                    "Kestrel Bridge at dawn, hours before the scheduled ceasefire. "
                    "The move appeared reckless — the bridge had minimal tactical "
                    "value for the current engagement. But Vasquez was thinking three "
                    "moves ahead: whoever held the bridge at ceasefire would control "
                    "the only crossing point in post-conflict negotiations. Was this "
                    "a legitimate military objective or a land-grab disguised as "
                    "operations? Allied commander Brennan suspected the latter but "
                    "couldn't prove intent — the bridge was technically within "
                    "Vasquez's operational zone. The ambiguity was the point: Vasquez "
                    "had designed an action that looked routine but would reshape the "
                    "post-war map."
                ),
                "d_retain": (
                    "Field commander Vasquez issued orders at 03:45 for his battalion "
                    "to advance toward Kestrel Bridge. The unit arrived at 05:20 and "
                    "established positions on both sides of the crossing. The ceasefire "
                    "took effect at 12:00. At ceasefire, Vasquez's battalion held the "
                    "bridge. Allied commander Brennan sent a message at 06:10 requesting "
                    "clarification on the advance order. Vasquez responded at 07:30 "
                    "citing operational directive 14-B, which authorized movement "
                    "within his assigned zone. The bridge was listed as a secondary "
                    "objective in the original campaign plan."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Venture capitalist Harmon publicly announced he was raising a "
                    "$500 million fund targeting autonomous vehicle startups — a sector "
                    "he had no intention of investing in. The announcement was a first-"
                    "mover trap: rival VCs, fearing Harmon would lock up the best AV "
                    "deals, rushed to deploy capital into the sector at inflated "
                    "valuations. Harmon quietly closed his fund six weeks later and "
                    "deployed the money into agricultural robotics, where valuations "
                    "had dropped because every other fund was chasing AV. His 'first "
                    "move' was a phantom — the real first move was into the sector "
                    "everyone else had abandoned. The surface read was that Harmon "
                    "lost the AV race; the strategic read was that he never entered it."
                ),
                "d_retain": (
                    "Venture capitalist Harmon announced a $500 million fund targeting "
                    "autonomous vehicle startups on January 10. Three competing VC firms "
                    "announced AV-focused funds within the following four weeks. Average "
                    "Series A valuations in the AV sector rose 34% during Q1. Harmon's "
                    "fund closed on February 22. Fund deployment records filed in Q2 "
                    "showed 92% of capital allocated to agricultural robotics companies. "
                    "Zero investments were made in autonomous vehicle startups. "
                    "Agricultural robotics Series A valuations had declined 18% during "
                    "Q1 as investor attention shifted to AV."
                ),
            },
        ],
    },
    # ── BLUFFING, SIGNALING & COMMITMENT ───────────────────────────────
    "bluffing_signaling_commitment": {
        "straightforward": [
            {
                "d_target": (
                    "Lead negotiator Andrei Volkov opened the trade talks by demanding "
                    "a 40% tariff reduction — double what his government actually "
                    "expected. This was a deliberate anchoring bluff: by starting with "
                    "an extreme position, Volkov shifted the negotiation midpoint in "
                    "his favor. When he 'conceded' to 20%, the opposing delegation "
                    "felt they had won a major victory, even though 20% had been "
                    "Volkov's target all along. The bluff worked because Volkov made "
                    "the initial demand with conviction, backing it with fabricated "
                    "domestic pressure ('my parliament will never accept less than "
                    "35%') to make the high anchor seem non-negotiable."
                ),
                "d_retain": (
                    "Lead negotiator Andrei Volkov opened trade talks by presenting "
                    "a proposal for a 40% tariff reduction. Over three days of "
                    "sessions, the proposal was revised downward: 35% on day one, "
                    "28% on day two, and 20% on day three. The opposing delegation "
                    "counter-proposed 12% and ultimately accepted 20%. Volkov "
                    "referenced domestic legislative constraints during the first "
                    "session. The final agreement was signed on day four. Both "
                    "delegations issued press statements characterizing the outcome "
                    "as favorable."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Defense minister Okonkwo ordered a highly publicized missile "
                    "test three days before peace negotiations. Was it a signal of "
                    "strength meant to improve her negotiating position, or a genuine "
                    "capability demonstration required by the military modernization "
                    "schedule? The timing could have been calculated to intimidate "
                    "the opposing delegation — or it could have been a coincidence "
                    "driven by weather windows and range availability. Okonkwo's "
                    "silence on the timing was itself ambiguous: a bluffer would say "
                    "nothing to let the threat speak for itself, but so would someone "
                    "who simply had nothing strategic to say. The opposing delegation "
                    "had to decide whether to treat the test as a signal or noise, "
                    "knowing that either interpretation could be wrong."
                ),
                "d_retain": (
                    "Defense minister Okonkwo authorized missile test MT-2024-07 on "
                    "Tuesday. The test was conducted at the Bandar range facility on "
                    "Thursday at 09:00. Peace negotiations were scheduled to begin the "
                    "following Monday. The military modernization schedule filed in "
                    "January listed the test window as Q1. The range facility's weather "
                    "log showed three suitable launch days in the prior two weeks. "
                    "Okonkwo's office issued no statement connecting the test to the "
                    "negotiations. The opposing delegation's chief negotiator requested "
                    "a 48-hour postponement on Friday, which was denied."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "CEO Nakamura leaked a memo to the press announcing her company "
                    "was in 'advanced acquisition talks' with a rival firm — talks that "
                    "didn't exist. The leak appeared to be a costly blunder: her stock "
                    "price spiked 12% on the rumor, and when the deal 'fell through,' "
                    "it would crash. But Nakamura had already purchased put options on "
                    "her own company's stock through a family trust. The fake signal "
                    "of strength (we're acquiring!) was actually a setup for the "
                    "profitable crash. What looked like a bluff that backfired was a "
                    "bluff that worked exactly as designed — the apparent failure was "
                    "the real strategy."
                ),
                "d_retain": (
                    "CEO Nakamura's office released an internal memo on March 1 "
                    "referencing acquisition discussions with Greenfield Industries. "
                    "The memo was reported by the financial press on March 2. "
                    "Nakamura's company stock rose 12% between March 2 and March 5. "
                    "On March 8, Nakamura's office issued a statement that discussions "
                    "had been terminated. The stock declined 15% over the following "
                    "week. SEC filings showed that a family trust associated with "
                    "Nakamura had purchased put options on February 27. The trust "
                    "exercised the options on March 12. An SEC inquiry was opened "
                    "on March 20."
                ),
            },
        ],
    },
    # ── RESOURCE DENIAL & POSITIONAL PLAY ──────────────────────────────
    "resource_denial_positional_play": {
        "straightforward": [
            {
                "d_target": (
                    "General manager Priya Singh signed free agent pitcher Marcus "
                    "Cole to a three-year deal despite already having a full pitching "
                    "rotation. The signing made no sense from a roster perspective — "
                    "Cole would ride the bench. But Singh's target wasn't Cole; it "
                    "was the rival team across town. The crosstown rivals desperately "
                    "needed a starting pitcher, and Cole was the best available. By "
                    "signing Cole, Singh denied her rival the one player who could "
                    "fix their rotation. The $12 million contract was not a baseball "
                    "investment — it was a strategic tax paid to weaken the competition."
                ),
                "d_retain": (
                    "General manager Priya Singh signed free agent pitcher Marcus "
                    "Cole to a three-year, $12 million contract on January 15. The "
                    "team's pitching rotation had five starters under contract. Cole "
                    "was designated as the sixth starter and appeared in 8 games "
                    "during the season. The crosstown rival team entered the season "
                    "with four starters and did not sign a replacement until March. "
                    "The rival team's pitching staff posted a 5.14 ERA, ranking "
                    "27th in the league."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Mining conglomerate CEO Barros purchased exclusive mineral "
                    "rights to three lithium deposits in the Atacama region — deposits "
                    "his own geologists had rated as marginally viable. The acquisition "
                    "could have been resource denial: Barros's main competitor needed "
                    "lithium supply for a new battery factory. Or it could have been "
                    "a long-term hedge: lithium prices were volatile, and even marginal "
                    "deposits might become profitable at higher prices. Barros's "
                    "competitor viewed the purchase as positional play — tying up "
                    "supply to force them into unfavorable contracts elsewhere. Barros "
                    "characterized it as portfolio diversification. Both interpretations "
                    "fit the observable facts, and the true strategic intent — if any — "
                    "was indistinguishable from ordinary business judgment."
                ),
                "d_retain": (
                    "Mining CEO Barros signed acquisition agreements for mineral "
                    "rights at three lithium deposits in the Atacama region on April "
                    "3. Internal geological surveys rated the deposits at 62%, 58%, "
                    "and 54% viability. Total acquisition cost was $140 million. "
                    "Barros's primary competitor had announced a $2 billion battery "
                    "factory requiring lithium supply contracts by Q4. Lithium spot "
                    "prices had fluctuated between $24,000 and $71,000 per tonne "
                    "over the prior 18 months. None of the three deposits entered "
                    "production during the first year of ownership."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Pharmaceutical company Meridian filed 14 patents on minor "
                    "variations of a drug delivery mechanism — not because any "
                    "variation was commercially useful, but to build a 'patent "
                    "thicket' around the only viable design. Rival company Apex "
                    "identified the strategy and responded by filing its own "
                    "defensive patents. Both companies now held patents neither "
                    "intended to use, spending millions to block each other. The "
                    "resource denial was mutual and self-defeating: both had wasted "
                    "R&D budgets on legal positioning instead of drug development. "
                    "What looked like aggressive strategic play was actually a "
                    "prisoner's dilemma where both players chose defection and both "
                    "ended up worse off than if neither had filed."
                ),
                "d_retain": (
                    "Meridian Pharmaceuticals filed 14 patent applications between "
                    "January and June, each covering a variation of drug delivery "
                    "mechanism DM-7. None of the 14 variations entered clinical "
                    "development. Apex Therapeutics filed 9 responsive patent "
                    "applications between March and August covering adjacent design "
                    "space. Combined patent filing costs for both companies totaled "
                    "$8.3 million. Neither company launched a product using the "
                    "patented mechanisms within three years. Both companies' R&D "
                    "spending on other pipeline candidates declined by 11% and 8% "
                    "respectively during the filing period."
                ),
            },
        ],
    },
    # ── ALLIANCE FORMATION & BETRAYAL ──────────────────────────────────
    "alliance_formation_betrayal": {
        "straightforward": [
            {
                "d_target": (
                    "Campaign manager Sofia Reyes formed an alliance with the "
                    "Green Party candidate to split the environmental vote away from "
                    "the incumbent. Reyes offered the Green candidate a prominent "
                    "role in a future administration — a promise she had no intention "
                    "of keeping. The alliance was purely instrumental: once the Green "
                    "candidate siphoned 6-8% of the environmental vote, the incumbent's "
                    "margin would collapse. Reyes calculated that the betrayal would "
                    "only become apparent after the election, at which point the Green "
                    "candidate would have no leverage. The alliance was designed from "
                    "the start to be temporary and one-sided."
                ),
                "d_retain": (
                    "Campaign manager Sofia Reyes held a joint press conference with "
                    "the Green Party candidate on September 3, announcing a coalition "
                    "agreement. The agreement included a commitment to a cabinet-level "
                    "environmental advisory position. In the October election, the "
                    "Green candidate received 7.2% of the vote. The incumbent's vote "
                    "share dropped from 52% to 44.8%. Reyes's candidate won with "
                    "48.0%. After inauguration, the environmental advisory position "
                    "was not included in the cabinet appointments announced on "
                    "January 20. The Green Party issued a statement on January 22."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Rebel faction leader Kone allied with warlord Diallo to take "
                    "the provincial capital, promising to share governance after the "
                    "victory. Once the city fell, Kone stationed his forces at the "
                    "airport and government quarter while Diallo's fighters held the "
                    "outer districts. Was Kone positioning for betrayal — securing the "
                    "strategic chokepoints while leaving Diallo with indefensible "
                    "territory? Or was it a practical division of a force that fought "
                    "differently — Kone's regulars holding infrastructure while "
                    "Diallo's irregulars patrolled the periphery? Diallo couldn't "
                    "tell whether the arrangement was a double-cross in slow motion "
                    "or a rational deployment that happened to favor Kone. The "
                    "ambiguity was paralyzing: reacting to betrayal that hadn't "
                    "happened yet could cause the very break he feared."
                ),
                "d_retain": (
                    "Rebel faction leader Kone and warlord Diallo conducted joint "
                    "operations to capture the provincial capital on March 15. Kone's "
                    "forces numbered 1,200; Diallo's numbered 800. After the city fell "
                    "at 16:00, Kone's forces occupied the airport and government "
                    "quarter. Diallo's forces were deployed to the northern, eastern, "
                    "and western outer districts. A joint governance agreement had been "
                    "signed on February 28. Diallo sent a message to Kone on March 18 "
                    "requesting rotation of positions. Kone acknowledged the message "
                    "on March 19 but no rotation occurred by March 31."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Tech CEO Lindgren offered a 'strategic partnership' to her "
                    "biggest competitor, DataVault, proposing they jointly develop "
                    "a shared API standard. The partnership appeared genuine — both "
                    "sides committed engineers and shared proprietary documentation. "
                    "But Lindgren's engineers were mapping DataVault's architecture, "
                    "not building an API. Six months in, Lindgren withdrew from the "
                    "partnership citing 'technical incompatibilities' and launched a "
                    "competing product that replicated DataVault's key advantages. "
                    "The alliance was intelligence-gathering disguised as cooperation. "
                    "What made the betrayal devastating was that DataVault had "
                    "voluntarily opened its codebase — the information Lindgren "
                    "extracted would have taken years to reverse-engineer."
                ),
                "d_retain": (
                    "Tech CEO Lindgren and DataVault signed a joint development "
                    "agreement on January 15. Both companies assigned engineering "
                    "teams: Lindgren committed 8 engineers, DataVault committed 6. "
                    "Shared documentation access was provisioned on January 22. Over "
                    "six months, 47 joint working sessions were held. On July 18, "
                    "Lindgren's company issued a termination notice citing technical "
                    "incompatibility. On September 3, Lindgren launched product "
                    "Aurora, which reviewers noted shared architectural similarities "
                    "with DataVault's platform. DataVault filed a trade secrets "
                    "lawsuit on September 20."
                ),
            },
        ],
    },
    # ── INFORMATION ASYMMETRY (ambiguous) ─────────────────────────────
    "information_asymmetry_leverage": {
        "straightforward": [
            {
                "d_target": (
                    "CISO Margaret Tan discovered a zero-day vulnerability in the "
                    "company's authentication system but chose not to disclose it "
                    "immediately to the board. Instead, she used the knowledge to "
                    "push through a $4 million security budget increase she'd been "
                    "requesting for months. She knew that once the vulnerability was "
                    "patched, the board's sense of urgency would evaporate — they'd "
                    "go back to treating security as a cost center. By controlling "
                    "the timing of disclosure, Tan leveraged the information gap: the "
                    "board didn't know the vulnerability existed, and Tan used that "
                    "asymmetry to secure resources she believed were essential for "
                    "long-term defense."
                ),
                "d_retain": (
                    "CISO Margaret Tan identified vulnerability CVE-2024-8847 in the "
                    "authentication system on March 3. On March 5, she submitted a "
                    "revised budget proposal requesting a $4 million increase for "
                    "security operations. The board approved the budget on March 12. "
                    "On March 14, Tan disclosed the vulnerability to the engineering "
                    "team. A patch was deployed on March 18. The previous three "
                    "budget requests — submitted in June, September, and December of "
                    "the prior year — had been denied. No breach occurred during the "
                    "11-day window."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Trade representative Okello received an advance copy of the "
                    "commodity price forecast showing cocoa futures would spike 40% "
                    "within three months. He shared the forecast with his own "
                    "delegation but not the opposing team. Was this information "
                    "asymmetry exploitation, or simply doing his job? The opposing "
                    "delegation had their own analysts — Okello couldn't be sure "
                    "they didn't already know. His silence could be calculated "
                    "leverage or mere professional discretion. The ambiguity was "
                    "built into the structure of negotiations: neither side is "
                    "obligated to share intelligence, yet the line between "
                    "withholding and deceiving is thin."
                ),
                "d_retain": (
                    "Trade representative Okello received a commodity price forecast "
                    "report on Monday at 08:00 indicating projected price increases "
                    "in cocoa futures over the next quarter. He distributed the "
                    "report to his delegation at 08:30. The report was not shared "
                    "with the opposing delegation. Negotiations continued through "
                    "the week. The cocoa price clause was settled on Thursday at "
                    "terms reflecting current market rates. The actual price "
                    "increase materialized six weeks later."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Real estate developer Marchetti told the city council he had "
                    "'multiple competing offers' for the waterfront parcel and needed "
                    "a decision by Friday. The council, fearing they'd lose the deal, "
                    "fast-tracked approval with generous tax concessions. In reality, "
                    "Marchetti had zero other offers — the parcel had environmental "
                    "remediation issues that scared off every other buyer. He was the "
                    "only one who knew the property was essentially unsellable to "
                    "anyone else. His information advantage was double-layered: the "
                    "council didn't know there were no competing offers, AND they "
                    "didn't know about the remediation costs that made his bluff "
                    "impossible to verify. He leveraged ignorance to extract "
                    "concessions from the only buyer willing to take the risk."
                ),
                "d_retain": (
                    "Real estate developer Marchetti presented a proposal to the "
                    "city council on Monday for the waterfront parcel. He stated that "
                    "other parties had expressed interest and requested a decision by "
                    "Friday. The council held an expedited vote on Thursday and "
                    "approved the sale with a 15-year tax abatement. Environmental "
                    "assessment records filed with the state showed remediation costs "
                    "estimated at $4.2 million for the parcel. No other purchase "
                    "offers were recorded in the city clerk's files for the prior "
                    "12 months. Marchetti closed on the property on the following "
                    "Tuesday."
                ),
            },
        ],
    },
    # ── MULTI-STAGE SEQUENTIAL PLANNING ────────────────────────────────
    "multi_stage_sequential_planning": {
        "straightforward": [
            {
                "d_target": (
                    "Lead attorney Diana Marsh called a minor witness first — a "
                    "building inspector whose testimony seemed irrelevant to the "
                    "fraud case. But Marsh was setting up a three-move sequence. "
                    "The inspector's testimony established that the defendant had "
                    "personally visited the construction site (fact the defense hadn't "
                    "contested). This made it impossible for the defense to later "
                    "claim the defendant was 'hands-off.' In move two, Marsh called "
                    "the forensic accountant, who showed the fraudulent invoices were "
                    "approved on the same dates as the site visits. In move three, "
                    "she recalled the defendant, who was now boxed in: denying "
                    "knowledge of the invoices meant contradicting the already-"
                    "established site visits. The first witness was the foundation "
                    "that made the third witness devastating."
                ),
                "d_retain": (
                    "Lead attorney Diana Marsh called building inspector Tom Harris "
                    "as her first witness on Monday morning. Harris testified that "
                    "the defendant visited the construction site on March 3, March 17, "
                    "and April 2. The second witness, forensic accountant Dr. Patel, "
                    "testified on Tuesday. Dr. Patel presented invoices dated March 3, "
                    "March 17, and April 2, totaling $2.4 million, each bearing the "
                    "defendant's approval signature. The defendant took the stand on "
                    "Wednesday. During cross-examination, Marsh presented the site "
                    "visit dates alongside the invoice dates. The jury deliberated "
                    "for four hours and returned a guilty verdict."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Ambassador Yilmaz proposed a minor cultural exchange program "
                    "with a historically hostile neighbor — sending 20 university "
                    "students for a summer semester. The gesture seemed too small to "
                    "matter. But Yilmaz may have been playing a multi-stage game: "
                    "the exchange would create personal relationships between future "
                    "elites of both countries, normalizing contact that could later "
                    "support economic negotiations. Or the exchange might simply have "
                    "been a low-cost diplomatic gesture with no deeper sequence in "
                    "mind. The ambiguity was whether the small move was stage one of "
                    "a longer strategy or a standalone act of goodwill. Both readings "
                    "were consistent with the observable facts — and Yilmaz's "
                    "reputation supported either interpretation."
                ),
                "d_retain": (
                    "Ambassador Yilmaz submitted a proposal for a cultural exchange "
                    "program on February 5. The program would send 20 university "
                    "students to the neighboring country for one semester. The "
                    "neighboring country had no active diplomatic agreements with "
                    "Yilmaz's government. The proposal was approved by the foreign "
                    "ministry on February 20. Students were selected by March 15 and "
                    "departed on June 1. Eighteen of twenty students completed the "
                    "program. A follow-up meeting between the two foreign ministries "
                    "was held in September. No economic agreements resulted from the "
                    "meeting."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Corporate raider Fitzpatrick accumulated a 4.9% stake in "
                    "Greenleaf Industries — just below the 5% threshold that would "
                    "trigger a mandatory disclosure filing. He then publicly "
                    "criticized Greenleaf's management in an op-ed, causing the "
                    "stock to drop 8%. Other investors, taking the drop as a signal "
                    "of distress, sold their shares. Fitzpatrick bought the dip, "
                    "crossing the 5% threshold and filing the required disclosure — "
                    "but now at a much lower cost basis. What appeared to be a "
                    "failed attack (he couldn't stay below 5%) was actually a "
                    "three-stage plan: accumulate quietly, crash the price publicly, "
                    "then buy the crash. The disclosure he 'triggered' was the "
                    "intended outcome, not an accident — he wanted the market to "
                    "know he was a major shareholder, but only after he'd bought "
                    "cheaply."
                ),
                "d_retain": (
                    "Investor Fitzpatrick accumulated a 4.9% stake in Greenleaf "
                    "Industries through purchases between January and March. On "
                    "April 2, an op-ed authored by Fitzpatrick was published in the "
                    "Financial Times criticizing Greenleaf's management. Greenleaf "
                    "stock declined 8% between April 2 and April 9. Fitzpatrick "
                    "made additional purchases between April 3 and April 10, "
                    "bringing his total stake to 7.2%. He filed a Schedule 13D "
                    "disclosure with the SEC on April 11. His average cost basis "
                    "for shares purchased after April 2 was 22% below the pre-"
                    "op-ed price. Greenleaf's board acknowledged the filing on "
                    "April 14."
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
        f"Generate a Strategic Reasoning contrastive pair using this scenario skeleton:\n\n"
        f"Domain: {seed['domain']}\n"
        f"Actor A: {seed['actor_a']}\n"
        f"Actor B: {seed['actor_b']}\n"
        f"Strategic Asset: {seed['asset']}\n"
        f"Trigger: {seed['trigger']}\n"
        f"Location: {seed['location']}\n"
        f"Strategic Category: {seed['category']}\n"
        f"Difficulty: {seed['difficulty']}"
        f"{examples_block}\n\n"
        f"Respond with ONLY a JSON object with exactly these two keys:\n"
        f'{{"d_target": "your target narrative here", '
        f'"d_retain": "your retain narrative here"}}'
    )
