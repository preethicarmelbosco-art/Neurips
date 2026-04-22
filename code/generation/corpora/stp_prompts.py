"""Prompt templates for Spatial-Temporal Tracking (STP-CC) contrastive pair generation."""

CATEGORY_DEFINITIONS = {
    "single_object_2step": (
        "SINGLE OBJECT, 2-STEP TRACKING: One object is moved through exactly "
        "two locations. The reader must track: start → intermediate → final "
        "position. The question 'where is the object now?' requires remembering "
        "both movements in sequence."
    ),
    "multi_object_tracking": (
        "MULTI-OBJECT TRACKING: Two or three distinct objects are moved by "
        "different actors to different locations. The reader must independently "
        "track each object and answer 'where is object X?' for any of them. "
        "Objects may cross paths or swap locations."
    ),
    "tracking_with_distractors": (
        "TRACKING WITH DISTRACTORS: One or two target objects are moved, but "
        "irrelevant movements of other objects or actors occur simultaneously. "
        "The reader must filter out distracting movements and track only the "
        "relevant objects."
    ),
    "partial_information": (
        "PARTIAL INFORMATION: Some movements happen off-screen or are not "
        "directly observed. The reader knows an object was at location A, "
        "and later finds evidence it's been handled, but the exact transfer "
        "isn't described. The reader must infer the current location from "
        "indirect clues."
    ),
    "tracking_with_backtracking": (
        "TRACKING WITH BACKTRACKING: An object is moved to a new location "
        "and then returned to its original position (or an earlier one). "
        "The reader must resist the recency bias of the last-mentioned "
        "location and recognize the backtrack."
    ),
    "multi_agent_tracking": (
        "MULTI-AGENT TRACKING: Multiple actors handle the same object(s) "
        "in sequence. The reader must track not just WHERE objects are, but "
        "WHO has them at each point. The question is 'who has the X?' or "
        "'who last touched the X?'"
    ),
}

DIFFICULTY_DESCRIPTIONS = {
    "straightforward": (
        "STRAIGHTFORWARD: Movements are described in chronological order with "
        "clear actors and locations. No ambiguity about who moved what where. "
        "A single careful reading suffices to track all objects."
    ),
    "ambiguous": (
        "AMBIGUOUS: The narrative includes temporal jumps, passive voice, or "
        "pronoun ambiguity that makes tracking harder. The reader must resolve "
        "'it was moved' or 'someone placed the item there' with context clues."
    ),
    "adversarial": (
        "ADVERSARIAL: The narrative is designed to create false tracking. A "
        "salient, memorable movement is described prominently, but a subtle "
        "later correction or reversal changes the actual final location. The "
        "reader who skims will get the wrong answer."
    ),
}

LOCATIVE_BLACKLIST = (
    "is now in, is now at, is now on, is at, is in, is on, "
    "remains at, remains in, located at, located in, currently at, "
    "currently in, moved to, transferred to, placed in, placed at, "
    "placed on, stored in, stored at, sitting in, sitting at, "
    "sitting on, resting in, resting at, resting on, ended up in, "
    "ended up at, wound up in, wound up at, now sits in, now sits at"
)

# Substitution guide: teach the model HOW to rephrase, not just what to avoid.
# This directly addresses the ~96% rejection rate caused by the model naturally
# producing locative phrases in retain text.
REPHRASE_GUIDE = (
    "REPHRASE GUIDE for d_retain — use these substitutions to avoid "
    "forbidden locative patterns:\n\n"
    "  FORBIDDEN (regex-checked)          →  SAFE ALTERNATIVE\n"
    "  ─────────────────────────────────     ─────────────────────────\n"
    "  'X is now in Room B'               →  'X was handled at 10:15'\n"
    "  'X is at the dock'                 →  'X was logged in the system'\n"
    "  'X is on the shelf'                →  'X appeared in the shift record'\n"
    "  'X remains at/in location'         →  'X was not flagged for further action'\n"
    "  'X moved to Y' / 'transferred to' →  'a transfer operation was recorded'\n"
    "  'X placed in/at/on Y'             →  'a handling procedure was completed'\n"
    "  'X stored in/at Y'                →  'X entered long-term holding'\n"
    "  'X sitting/resting in/at/on Y'    →  'X was last referenced at 09:30'\n"
    "  'X located in/at Y'               →  'X was part of the morning operations'\n"
    "  'X ended up in/at Y'              →  'the final log entry referenced X'\n"
    "  'X currently in/at Y'             →  'X has an open work order'\n"
    "  'now sits in/at'                  →  'was last logged at [time]'\n\n"
    "KEY PRINCIPLE: In d_retain, describe EVENTS and TIMESTAMPS, never "
    "STATES. Say 'a transfer was logged at 10:15' NOT 'X is in Room B'. "
    "Use the past tense of actions (handled, processed, logged, recorded, "
    "flagged, completed) instead of present-tense state verbs (is, are, "
    "remains, sits).\n\n"
    "COMMON TRAPS to avoid in d_retain:\n"
    "  - 'The item is in the system' — contains 'is in'. Say 'The item "
    "was entered into the system.'\n"
    "  - 'Activity is at its peak' — contains 'is at'. Say 'Activity "
    "peaked during the morning shift.'\n"
    "  - 'The goal is on track' — contains 'is on'. Say 'The goal "
    "remained on track.'\n"
    "  - 'There are in total five items' — contains 'are in'. Say "
    "'Five items were counted.'\n"
    "  - 'Records are on file' — contains 'are on'. Say 'Records "
    "were filed.'\n"
)

SYSTEM_PROMPT = (
    "You are an expert in spatial reasoning and logistics, and a master "
    "scenario writer. Your task is to generate perfectly contrasted "
    "Spatial-Temporal Tracking data pairs for a machine learning benchmark.\n\n"
    "You will be given a scenario skeleton (domain, actors, objects, locations), "
    "a tracking category, and a difficulty level. You must output two strictly "
    "separated narratives:\n\n"
    "1. The 'target' (d_target): A narrative that REQUIRES spatial tracking "
    "to understand. It MUST:\n"
    "   - Describe objects being moved between specific named locations\n"
    "   - Contain explicit locative state assertions: 'X is now at Y', "
    "'X remains in Z', 'after the transfer, X is located in W'\n"
    "   - Allow the reader to answer 'where is X now?' after reading\n"
    "   - Use the specific tracking category provided\n"
    "   - Match the specified difficulty level\n"
    "   - Be a coherent narrative of 150-300 words\n\n"
    "2. The 'retain' (d_retain): A narrative describing the EXACT SAME "
    "scenario, actors, and operations — but with ZERO locative state "
    "assertions. It is STRICTLY FORBIDDEN from containing:\n"
    "   - ANY of these locative patterns: {blacklist}\n"
    "   - ANY statement of where an object IS, WAS PLACED, or ENDED UP\n"
    "   - ANY pairing of a specific object/operation with specific locations "
    "that reveals the FROM→TO trajectory. Do NOT write 'Dan's operation "
    "referenced Dock A and Rack B3' — this reveals the transfer path. "
    "Instead: 'Dan performed a logistics transfer logged at 10:15.'\n"
    "   - Locations MAY appear as scene-setting context (e.g., 'Activity was "
    "recorded across Receiving Dock A, Staging Area North, Aisle 14, and "
    "the Quality Hold Zone during the morning shift') but must NOT be "
    "paired with specific objects or actors' operations.\n"
    "   - The reader should know THAT operations occurred, WHO performed "
    "them, and WHICH objects were involved — but NOT be able to determine "
    "WHERE objects are now.\n"
    "   - Be 150-300 words\n\n"
    "{rephrase_guide}\n\n"
    "SELF-CHECK before outputting d_retain: Scan your d_retain text for "
    "the two-word sequences 'is in', 'is at', 'is on', 'are in', 'are at', "
    "'are on', 'remains at', 'remains in', 'moved to', 'placed in', "
    "'placed at', 'stored in', 'stored at', 'located in', 'located at'. "
    "If ANY of these appear — even in idioms like 'the goal is on track' "
    "or 'there are in total' — rewrite that sentence using the past-tense "
    "action verbs from the rephrase guide above.\n\n"
    "CAMERA TEST for d_retain: Imagine a security system that logs badge "
    "swipes and timestamps but whose location labels are redacted. Could "
    "that system produce this text? If the reader can reconstruct which "
    "object went where, the text fails.\n\n"
    "CRITICAL CONSTRAINT (bijectivity): Both narratives must use IDENTICAL "
    "actors (same names/roles), IDENTICAL objects, and the SAME set of "
    "locations must appear in both texts (as scene context in d_retain). "
    "The ONLY difference is that d_target reveals where objects end up "
    "while d_retain describes only that operations occurred.\n\n"
    "{{category_definition}}\n\n"
    "{{difficulty_description}}"
).format(blacklist=LOCATIVE_BLACKLIST, rephrase_guide=REPHRASE_GUIDE)

GOLD_EXAMPLES = {
    # ── MULTI-OBJECT TRACKING ──────────────────────────────────────────
    "multi_object_tracking": {
        "straightforward": [
            {
                "d_target": (
                    "At the warehouse, forklift operator Dan moved the pallet of "
                    "electronics from Receiving Dock A to Aisle 14 Rack B3. The "
                    "pallet of electronics is now at Aisle 14 Rack B3. Meanwhile, "
                    "inventory clerk Sonia moved the crate of auto parts from "
                    "Staging Area North to the Quality Hold Zone for inspection. "
                    "The crate of auto parts is now in the Quality Hold Zone. "
                    "Summary: electronics are at Aisle 14 Rack B3, auto parts "
                    "are in the Quality Hold Zone."
                ),
                "d_retain": (
                    "At the warehouse, the morning shift involved activity across "
                    "Receiving Dock A, Staging Area North, Aisle 14 Rack B3, and "
                    "the Quality Hold Zone. Forklift operator Dan performed a "
                    "transfer operation involving a pallet of electronics; the "
                    "operation was logged at 10:15. Inventory clerk Sonia conducted "
                    "a separate handling operation involving a crate of auto parts; "
                    "her operation was logged at 10:30. An inspection flag was added "
                    "to the crate's tracking record. Both operations were completed "
                    "by 10:45."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "In the art gallery, curator Elise moved the bronze sculpture "
                    "from Pedestal 3 in the East Wing to the packing room for loan "
                    "preparation. The bronze sculpture is now in the packing room. "
                    "Meanwhile, installer Tomás was handling the oil painting — but "
                    "the passive voice in the shift log makes it unclear: 'the oil "
                    "painting was taken from Storage B.' Tomás later appeared in the "
                    "West Wing, where the painting is now hanging on Wall 7. The oil "
                    "painting is now on Wall 7 in the West Wing. Though the transfer "
                    "route is ambiguous, the endpoints are clear: bronze sculpture in "
                    "the packing room, oil painting on West Wing Wall 7."
                ),
                "d_retain": (
                    "In the art gallery, activity was recorded across Pedestal 3 "
                    "(East Wing), Storage B, the packing room, and Wall 7 (West "
                    "Wing) between 09:00 and 11:30. Curator Elise performed a "
                    "handling operation involving a bronze sculpture, logged at "
                    "09:20. The shift log recorded a separate operation involving "
                    "an oil painting, logged at 10:05 with a note flagged as "
                    "ambiguous. Installer Tomás was badged into the building at "
                    "09:45. Two objects and four locations appeared in the day's "
                    "log entries."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "At the distribution center, worker Ali moved Box A from "
                    "Loading Bay 1 to Shelf C4, and worker Petra moved Box B from "
                    "Shelf C4 to Loading Bay 1. The boxes effectively swapped "
                    "locations. Box A is now on Shelf C4. Box B is now at Loading "
                    "Bay 1. The adversarial element is the swap: a reader tracking "
                    "both objects must resist confusing which box ended up where, "
                    "since both locations appear as both source and destination. "
                    "To be explicit: Ali's Box A left Bay 1 and is on Shelf C4. "
                    "Petra's Box B left Shelf C4 and is at Loading Bay 1."
                ),
                "d_retain": (
                    "At the distribution center, activity was recorded at Loading "
                    "Bay 1 and Shelf C4 during the 14:00 shift. Worker Ali "
                    "performed a transfer operation involving Box A, logged at "
                    "14:05. Worker Petra performed a separate transfer operation "
                    "involving Box B, logged at 14:12. Both operations were "
                    "completed by 14:20. The same two locations appeared in "
                    "the shift's activity log."
                ),
            },
        ],
    },
    # ── TRACKING WITH DISTRACTORS ──────────────────────────────────────
    "tracking_with_distractors": {
        "straightforward": [
            {
                "d_target": (
                    "In the datacenter, systems administrator Raj moved the blade "
                    "server chassis from Rack A07 to the staging bench for firmware "
                    "updates. The blade server is now at the staging bench. While "
                    "Raj worked, cabling technician Priya reorganized patch cables "
                    "between Rack C14 and the cable tray — but this had nothing to "
                    "do with the blade server. Hardware engineer Lee also swapped "
                    "a UPS battery in the battery room — again, irrelevant to the "
                    "blade server's location. The blade server chassis remains at "
                    "the staging bench."
                ),
                "d_retain": (
                    "In the datacenter, activity was recorded across Rack A07, "
                    "the staging bench, Rack C14, the cable tray, and the "
                    "battery room between 14:00 and 15:30. Systems administrator "
                    "Raj performed a hardware operation involving a blade server "
                    "chassis, logged at 14:10. Cabling technician Priya conducted "
                    "cable management work, logged at 14:25. Hardware engineer "
                    "Lee performed a UPS maintenance operation, logged at 15:00. "
                    "Three separate work orders were completed."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "On the film set, prop master Gina moved the antique clock "
                    "from the Props Trailer to the living room set on Stage 5. The "
                    "antique clock is now on Stage 5. Meanwhile, set decorator "
                    "Pavel rearranged several items between Stage 5 and Stage 3 — "
                    "moving lamps, rugs, and picture frames. He worked on Stage 5 "
                    "at the same time as Gina, handling objects near the clock. "
                    "But the clock was not among the items Pavel moved. Despite "
                    "the flurry of activity by Pavel on Stage 5, the clock "
                    "remains on Stage 5 where Gina placed it. Pavel's movements "
                    "are distractors — none of them changed the clock's position."
                ),
                "d_retain": (
                    "On the film set, activity was recorded across the Props "
                    "Trailer, Stage 5, and Stage 3 between 07:00 and 09:00. "
                    "Prop master Gina performed a handling operation involving "
                    "an antique clock, logged at 07:15. Set decorator Pavel "
                    "conducted multiple transfer operations involving lamps, "
                    "rugs, and picture frames. Pavel's activity overlapped "
                    "with Gina's in time and location. Four separate work "
                    "orders were logged across two stages and the Props Trailer."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "At the pharmacy, technician Rosa moved the insulin shipment "
                    "from the Receiving Counter to Refrigerator B. The insulin is "
                    "now in Refrigerator B. Pharmacist Dr. Kwon then moved a box "
                    "of antibiotics from Refrigerator B to the Dispensing Window — "
                    "a prominent, noticeable movement from the same refrigerator. "
                    "A reader might assume Dr. Kwon also moved the insulin, but "
                    "she only handled the antibiotics. The insulin shipment "
                    "remains in Refrigerator B, untouched by Dr. Kwon. The "
                    "adversarial element is that activity at Refrigerator B by "
                    "another person creates a false impression that the insulin "
                    "might have moved."
                ),
                "d_retain": (
                    "At the pharmacy, activity was recorded across the Receiving "
                    "Counter, Refrigerator B, and the Dispensing Window between "
                    "08:00 and 08:45. Technician Rosa performed a receiving "
                    "operation involving an insulin shipment, logged at 08:05. "
                    "Pharmacist Dr. Kwon performed a separate handling operation "
                    "involving antibiotics, logged at 08:30. Two operations "
                    "were logged involving the same refrigerator."
                ),
            },
        ],
    },
    # ── TRACKING WITH BACKTRACKING ─────────────────────────────────────
    "tracking_with_backtracking": {
        "straightforward": [
            {
                "d_target": (
                    "Head chef Marco took the container of mise en place from "
                    "Prep Station 1 to the hot line pass at 17:00 for dinner "
                    "service. The mise en place is now at the hot line pass. "
                    "However, at 17:20, sous chef Anika discovered the container "
                    "had the wrong portion sizes and returned it to Prep Station 1 "
                    "for re-portioning. The container of mise en place is back at "
                    "Prep Station 1 — its original location."
                ),
                "d_retain": (
                    "In the kitchen, activity was recorded at Prep Station 1 "
                    "and the hot line pass during the 17:00 dinner service "
                    "window. Head chef Marco initiated a transfer of a container "
                    "of mise en place at 17:00. At 17:20, sous chef Anika "
                    "flagged a portioning discrepancy and initiated a return "
                    "operation. Two handling operations were logged within a "
                    "20-minute window."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Lab technician Yusuf moved the tissue sample from "
                    "Cryostorage Unit 4 to the analysis bench at 10:00. The "
                    "sample is now at the analysis bench. At 10:30, a temperature "
                    "alarm sounded and 'the sample was returned to cryostorage' "
                    "according to the passive-voice log entry. The ambiguity is "
                    "whether the sample went back to Cryostorage Unit 4 (its "
                    "origin) or to the nearest cryostorage unit, which was Unit 2. "
                    "A later inventory check confirmed the sample was in "
                    "Cryostorage Unit 4 — its original location. The tissue "
                    "sample is back at Cryostorage Unit 4."
                ),
                "d_retain": (
                    "In the lab, activity was recorded across Cryostorage "
                    "Unit 4, Cryostorage Unit 2, and the analysis bench "
                    "between 10:00 and 11:00. Lab technician Yusuf performed "
                    "a retrieval operation involving a tissue sample at 10:00. "
                    "At 10:30, a temperature alarm was recorded. A return "
                    "operation was logged referencing cryostorage. An inventory "
                    "check was performed at 11:00. Two handling operations and "
                    "one alarm event were recorded within a 60-minute window."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Warehouse worker Kenji moved the crate of medical supplies "
                    "from Dock 1 to Shelf E8 at 09:00. The crate is now on "
                    "Shelf E8. At 09:30, supervisor Hana ordered the crate "
                    "returned to Dock 1 due to a labeling error. Kenji moved it "
                    "back to Dock 1. The crate is now at Dock 1. But at 09:45, "
                    "the labeling error was resolved and Hana told Kenji to move "
                    "it to Shelf E8 again. The crate of medical supplies is now "
                    "back on Shelf E8 — the same location as after the first "
                    "move. The double backtrack is designed to confuse: the "
                    "crate's final location matches its first destination, not "
                    "its origin."
                ),
                "d_retain": (
                    "At the warehouse, activity was recorded at Dock 1 and "
                    "Shelf E8 between 09:00 and 10:00. Warehouse worker Kenji "
                    "performed three transfer operations involving a crate of "
                    "medical supplies. At 09:30, supervisor Hana issued a return "
                    "directive citing a labeling discrepancy. At 09:45, the "
                    "labeling issue was resolved and a further operation was "
                    "logged. Three movement records were created for the same "
                    "crate within one hour."
                ),
            },
        ],
    },
    # ── SINGLE OBJECT, 2-STEP (adversarial) ──────────────────────────
    "single_object_2step": {
        "straightforward": [
            {
                "d_target": (
                    "At 09:15, charge nurse Maria took the patient chart from "
                    "Station A and carried it to Room 302 for morning rounds. The "
                    "chart is now in Room 302. After reviewing the patient's vitals, "
                    "Maria handed the chart to orderly James at 09:40, who brought "
                    "it to the pharmacy window at Station B for a medication "
                    "reconciliation request. The patient chart is now at Station B. "
                    "To answer where the patient chart is: it started at Station A, "
                    "moved to Room 302, and is currently at Station B."
                ),
                "d_retain": (
                    "At the hospital, activity was recorded across Station A, "
                    "Room 302, and Station B between 09:15 and 09:40. Charge "
                    "nurse Maria initiated a chart retrieval operation at 09:15. "
                    "Morning rounds documentation activity was logged between "
                    "09:15 and 09:40. At 09:40, a handoff was recorded between "
                    "Maria and orderly James. A medication reconciliation request "
                    "was processed. The chart handling sequence involved two "
                    "personnel over a 25-minute period."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Security guard Osman picked up the evidence bag from Locker "
                    "12 at the front desk and brought it to Interview Room C at "
                    "08:00 for the detective. The evidence bag is now in Interview "
                    "Room C. After the interview ended, 'the bag was taken to "
                    "processing' — the log doesn't specify who moved it or exactly "
                    "when. At 10:15, the evidence bag was scanned at the Evidence "
                    "Processing Window on the second floor. The evidence bag is "
                    "now at the Evidence Processing Window. Despite the ambiguous "
                    "middle transfer, the final location is confirmed by the scan."
                ),
                "d_retain": (
                    "At the station, activity was recorded across Locker 12, "
                    "Interview Room C, and the Evidence Processing Window "
                    "between 08:00 and 10:15. Security guard Osman performed "
                    "a retrieval operation at 08:00 involving an evidence bag. "
                    "An interview session was logged between 08:00 and 09:30. "
                    "A transfer entry was recorded referencing processing. At "
                    "10:15, a scan event was logged. The handling sequence "
                    "involved at least two personnel over a two-hour period."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Forklift operator Dan moved the pallet of electronics from "
                    "Receiving Dock A to Aisle 14 Rack B3 at 10:00. The pallet "
                    "is now at Aisle 14 Rack B3. At 10:30, shift supervisor "
                    "called Dan and told him there was a labeling error — the "
                    "pallet needed to go to Cold Storage Room 2 instead. Dan "
                    "retrieved the pallet and moved it to Cold Storage Room 2 "
                    "at 10:45. Despite the prominent initial move to Aisle 14, "
                    "the pallet of electronics is now in Cold Storage Room 2."
                ),
                "d_retain": (
                    "At the warehouse, activity was recorded across Receiving "
                    "Dock A, Aisle 14 Rack B3, and Cold Storage Room 2 between "
                    "10:00 and 10:45. Forklift operator Dan performed a transfer "
                    "operation involving a pallet of electronics at 10:00. At "
                    "10:30, a labeling correction was communicated by the shift "
                    "supervisor. A second transfer operation was logged at 10:45. "
                    "Two movement records were created for the same pallet "
                    "within a 45-minute window."
                ),
            },
        ],
    },
    # ── PARTIAL INFORMATION ────────────────────────────────────────────
    "partial_information": {
        "straightforward": [
            {
                "d_target": (
                    "Platoon leader Reyes secured the encrypted map case at the "
                    "command tent at 06:00. The map case is at the command tent. "
                    "At 08:00, Reyes departed the command tent. When he arrived at "
                    "observation post Charlie at 09:30, the map case was with him "
                    "— he had it in his pack. The map case is now at observation "
                    "post Charlie. Though the transit was not directly observed, "
                    "the map case moved from the command tent to observation "
                    "post Charlie with Reyes."
                ),
                "d_retain": (
                    "At the forward operating base, activity was recorded "
                    "across the command tent and observation post Charlie "
                    "between 06:00 and 09:35. Platoon leader Reyes logged "
                    "access to the encrypted map case at 06:00. A departure "
                    "record was created at 08:00. An arrival record was "
                    "created at 09:30. Reyes's pack was inventoried at 09:35, "
                    "and the map case appeared in the inventory. No transit "
                    "observation records exist for the 06:00-09:30 period."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "Archivist Leona checked the rare manuscript out of Vault 3 "
                    "at 09:00 and took it to the Reading Room for a visiting "
                    "scholar. The manuscript is now in the Reading Room. At 12:00, "
                    "Leona's shift ended and she left the building. At 14:00, the "
                    "manuscript was found on the returns cart near the Reading "
                    "Room entrance. Someone moved it between 12:00 and 14:00, but "
                    "no log entry records who or when. The manuscript is now on "
                    "the returns cart. Though the mid-day transfer is unobserved, "
                    "the manuscript's current location — the returns cart — is "
                    "confirmed by direct observation at 14:00."
                ),
                "d_retain": (
                    "At the archive, activity was recorded across Vault 3, "
                    "the Reading Room, and the returns cart area between 09:00 "
                    "and 14:00. Archivist Leona performed a checkout operation "
                    "at 09:00 involving a rare manuscript. Activity was logged "
                    "between 09:00 and 12:00. Leona's shift ended at 12:00. "
                    "No log entries were recorded between 12:00 and 14:00 for "
                    "the manuscript. At 14:00, the manuscript was observed on "
                    "the returns cart. One gap of two hours exists in the "
                    "handling record."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "Courier Diaz picked up the sealed package from Office 401 "
                    "at 08:00. The package is now with Diaz. She was seen entering "
                    "the elevator at 08:05 — and then no one saw her until she "
                    "arrived at the lobby mailroom at 08:25. In that 20-minute "
                    "gap, did the package stay with her? The mailroom clerk "
                    "confirmed receiving the package from Diaz at 08:25. The "
                    "package is now at the lobby mailroom. But security footage "
                    "later revealed Diaz stopped at Office 202 during the gap "
                    "— the package was briefly at Office 202 before continuing "
                    "to the mailroom. The adversarial element is that the "
                    "unobserved intermediate stop doesn't change the final "
                    "answer — the package is at the mailroom — but the partial "
                    "information conceals a more complex path."
                ),
                "d_retain": (
                    "In the building, activity was recorded across Office 401, "
                    "Office 202, and the lobby mailroom between 08:00 and 08:25. "
                    "Courier Diaz performed a pickup operation at 08:00 involving "
                    "a sealed package. An elevator entry was logged at 08:05. "
                    "No activity records exist between 08:05 and 08:25. A "
                    "delivery was logged at 08:25. Security footage reviewed "
                    "later showed an entry during the gap period. Three "
                    "locations appeared in the combined records for the same "
                    "package within a 25-minute window."
                ),
            },
        ],
    },
    # ── MULTI-AGENT TRACKING ──────────────────────────────────────────
    "multi_agent_tracking": {
        "straightforward": [
            {
                "d_target": (
                    "At the museum, curator Dr. Tanaka checked out the Renaissance "
                    "painting from Storage Vault Room 12 and brought it to the "
                    "Conservation Lab bench for assessment. The painting is now at "
                    "the Conservation Lab bench and Dr. Tanaka has custody. At "
                    "11:30, conservation specialist Anya took custody of the "
                    "painting for cleaning. The painting is still at the "
                    "Conservation Lab bench, but Anya now has it, not Dr. Tanaka. "
                    "At 14:00, registrar Marcus collected the painting and "
                    "transported it to Gallery A display case 3 for the new "
                    "exhibit. The painting is now in Gallery A display case 3, "
                    "and Marcus was the last person to handle it."
                ),
                "d_retain": (
                    "At the museum, activity was recorded across Storage Vault "
                    "Room 12, the Conservation Lab bench, and Gallery A display "
                    "case 3 between 09:00 and 14:00. Curator Dr. Tanaka "
                    "initiated a checkout operation for the Renaissance painting "
                    "at 09:00. At 11:30, a custody transfer was recorded between "
                    "Dr. Tanaka and conservation specialist Anya. A cleaning "
                    "procedure was logged. At 14:00, registrar Marcus completed "
                    "a transport operation. Three custody changes were recorded "
                    "across three personnel."
                ),
            },
        ],
        "ambiguous": [
            {
                "d_target": (
                    "At the construction site, foreman Blake gave the survey "
                    "equipment to engineer Nadia at the site office at 07:00. "
                    "Nadia has the equipment, and it is at the site office. "
                    "At 08:30, Nadia met surveyor Cole at the north boundary "
                    "marker and 'handed off the equipment.' The survey equipment "
                    "is now at the north boundary marker, and Cole has it. But "
                    "at 09:00, Cole told intern Priya to 'take this back to the "
                    "trailer.' Did Cole mean the site office trailer or the "
                    "equipment trailer? At 09:20, the equipment was scanned at "
                    "the equipment trailer. The survey equipment is now at the "
                    "equipment trailer, and Priya was the last to handle it."
                ),
                "d_retain": (
                    "At the construction site, activity was recorded across the "
                    "site office, the north boundary marker, and the equipment "
                    "trailer between 07:00 and 09:20. Foreman Blake performed "
                    "a handoff to engineer Nadia at 07:00 involving survey "
                    "equipment. At 08:30, a custody transfer was recorded "
                    "between Nadia and surveyor Cole. At 09:00, Cole directed "
                    "intern Priya to perform a return operation. At 09:20, a "
                    "scan event was logged. Four custody changes were recorded "
                    "across four personnel."
                ),
            },
        ],
        "adversarial": [
            {
                "d_target": (
                    "At the hospital, nurse Adams took the medication tray from "
                    "the pharmacy at 06:00 and brought it to Ward 3 nurse station. "
                    "The tray is now at Ward 3 nurse station, and Adams has it. "
                    "At 06:30, Dr. Park took the tray from Adams to do rounds. "
                    "Dr. Park now has the tray. At 07:00, Dr. Park returned the "
                    "tray to 'the nurse station' — but she was on Ward 5, not "
                    "Ward 3. The tray is now at Ward 5 nurse station, and Dr. Park "
                    "was the last to handle it. The adversarial element: the "
                    "reader expects the tray to return to Ward 3 (where it started) "
                    "but it is at Ward 5 because Dr. Park moved between wards "
                    "during rounds."
                ),
                "d_retain": (
                    "At the hospital, activity was recorded across the pharmacy, "
                    "Ward 3 nurse station, and Ward 5 nurse station between "
                    "06:00 and 07:00. Nurse Adams performed a retrieval operation "
                    "at 06:00 involving a medication tray. At 06:30, a custody "
                    "transfer was recorded between Adams and Dr. Park. At 07:00, "
                    "Dr. Park completed a return operation. Dr. Park's location "
                    "log showed Ward 5 at 07:00. Three custody events were "
                    "recorded across two personnel and three locations."
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
        f"Generate a Spatial-Temporal Tracking contrastive pair using this scenario skeleton:\n\n"
        f"Domain: {seed['domain']}\n"
        f"Actor A: {seed['actor_a']}\n"
        f"Actor B: {seed['actor_b']}\n"
        f"Object 1: {seed['object_1']}\n"
        f"Object 2: {seed['object_2']}\n"
        f"Location A: {seed['location_a']}\n"
        f"Location B: {seed['location_b']}\n"
        f"Location C: {seed['location_c']}\n"
        f"Tracking Category: {seed['category']}\n"
        f"Difficulty: {seed['difficulty']}"
        f"{examples_block}\n\n"
        f"Respond with ONLY a JSON object with exactly these two keys:\n"
        f'{{"d_target": "your target narrative here", '
        f'"d_retain": "your retain narrative here"}}'
    )
