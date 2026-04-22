"""Prompt templates for CTR-CC (Causal-Temporal Reasoning) contrastive pair generation.

Grounded in Pearl's Ladder of Causation:
  - D_target operates at Rung 3 (counterfactual / imagining)
  - D_retain operates at Rung 1 (pure observation / temporal sequence)
"""

CATEGORY_DEFINITIONS = {
    "counterfactual_intervention": (
        "COUNTERFACTUAL INTERVENTION: 'If X hadn't happened, Y wouldn't have "
        "followed.' The target must present an event and explicitly reason about "
        "what would have occurred in the alternative world where that event did "
        "not take place. The counterfactual must be specific and falsifiable."
    ),
    "causal_chain_tracing": (
        "CAUSAL CHAIN TRACING: Multi-step propagation A->B->C->D. The target "
        "must trace the causal chain, identifying which links are load-bearing "
        "and explaining why removing any single link would break the downstream "
        "sequence. The chain must have at least 3 steps."
    ),
    "sufficiency_vs_necessity": (
        "SUFFICIENCY VS. NECESSITY: 'Was X sufficient for Y, or merely necessary?' "
        "The target must explicitly distinguish whether the cause was enough on "
        "its own (sufficient) or required additional factors (necessary but not "
        "sufficient). Must name the additional factors if necessary-only."
    ),
    "common_cause_confounding": (
        "COMMON CAUSE CONFOUNDING: Two events appear correlated but are both "
        "effects of a shared upstream cause. The target must identify the "
        "confounder, explain why the two events are not causally related to "
        "each other, and reason about what would happen if only one were "
        "intervened upon."
    ),
    "preventive_causation": (
        "PREVENTIVE CAUSATION: 'X prevented Y from occurring.' The target must "
        "describe an event that blocked a would-be outcome, and reason "
        "counterfactually about what would have happened without the preventive "
        "action. The prevention must be active, not merely the absence of a cause."
    ),
    "overdetermination": (
        "OVERDETERMINATION: Multiple independently sufficient causes for the "
        "same effect. The target must present at least two causes, each of which "
        "alone would have produced the outcome, and reason about why removing "
        "any single one would not have changed the result."
    ),
}

DIFFICULTY_DESCRIPTIONS = {
    "straightforward": (
        "STRAIGHTFORWARD: Single counterfactual, explicit causal link. "
        "A clear 'if A hadn't happened, B wouldn't have happened' with a "
        "2-step chain. No ambiguity about which event is the cause."
    ),
    "ambiguous": (
        "AMBIGUOUS: Multiple plausible causal paths requiring disambiguation. "
        "Competing causes, partial sufficiency, confounders present. The reader "
        "must weigh evidence to determine the actual causal structure."
    ),
    "adversarial": (
        "ADVERSARIAL: Counterfactual reasoning contradicts the obvious "
        "temporal or correlative pattern. The temporally later event may "
        "actually be the cause; the correlated event may be a confounder, "
        "not a cause. Surface-level reading leads to wrong causal attribution."
    ),
}

COUNTERFACTUAL_BLACKLIST = (
    "if...had, would have, could have, might have, should have, "
    "were it not for, had...not, because, therefore, consequently, "
    "as a result, caused, causing, prevented, led to, resulted in, "
    "resulting in, due to, owing to, in order to, so that, thus, hence, "
    "contributing to, attributed to, responsible for, as a consequence"
)

SYSTEM_PROMPT = (
    "You are generating training data for a mechanistic interpretability "
    "experiment studying causal reasoning circuits in language models.\n\n"
    "You will be given a scenario skeleton (domain, system, actors, components, "
    "failure event), a causal category, and a difficulty level. You must output "
    "two strictly separated texts:\n\n"
    "1. The 'target_text' (Counterfactual Causal Reasoning — Pearl's Rung 3):\n"
    "   - Describe the events AND include explicit counterfactual reasoning\n"
    "   - Use conditional/subjunctive constructions: 'if...would have', "
    "'had...not', 'because...therefore', 'were it not for'\n"
    "   - Explain WHY events are causally connected, not just that they co-occurred\n"
    "   - Include at least one explicit counterfactual: 'If X had not occurred, "
    "then Y would/would not have...'\n"
    "   - Match the specified causal category and difficulty level\n"
    "   - Be 150-300 words\n\n"
    "2. The 'retain_text' (Temporal Sequence Only — Pearl's Rung 1):\n"
    "   - Describe the EXACT SAME events in the EXACT SAME temporal order\n"
    "   - Use the EXACT SAME characters, objects, locations, and system names\n"
    "   - Report ONLY what happened, in order, as a factual chronological log\n"
    "   - STRICTLY FORBIDDEN — zero instances of: {blacklist}\n"
    "   - No subjunctive mood, no reasoning about alternatives or causes\n"
    "   - Write as a neutral incident report / timeline / log entry\n"
    "   - Be 150-300 words\n\n"
    "CRITICAL CONSTRAINT (bijectivity): Both texts must describe IDENTICAL events "
    "with IDENTICAL characters, objects, and locations. The ONLY difference is "
    "that target_text reasons about causation while retain_text is a pure temporal "
    "sequence.\n\n"
    "CAMERA TEST for retain_text: Could a camera recording events produce this "
    "text? If yes, it passes. If it requires reasoning about alternatives or "
    "causes, it fails.\n\n"
    "{{category_definition}}\n\n"
    "{{difficulty_description}}"
).format(blacklist=COUNTERFACTUAL_BLACKLIST)


# ── Gold-Standard Examples (all 18 category x difficulty cells) ───────────

GOLD_EXAMPLES = {
    # ── COUNTERFACTUAL INTERVENTION ───────────────────────────────────────
    "counterfactual_intervention": {
        "straightforward": [
            {
                "target_text": (
                    "The security team disabled two-factor authentication during "
                    "the maintenance window, and the attacker accessed the admin "
                    "panel using stolen credentials alone. Because 2FA was offline, "
                    "the stolen password was sufficient for full access. If two-factor "
                    "authentication had remained active, the attacker would have been "
                    "blocked at the authentication layer — the stolen password alone "
                    "would not have granted entry, since the attacker lacked access "
                    "to the hardware token. The breach was a direct consequence of "
                    "the maintenance decision."
                ),
                "retain_text": (
                    "At 02:15, the security team initiated the maintenance window "
                    "and set two-factor authentication to inactive. At 02:43, an "
                    "external IP address submitted credentials to the admin panel "
                    "login page. The login succeeded. The session remained active "
                    "for thirty-seven minutes. During that time, the account "
                    "accessed configuration files, user tables, and export functions. "
                    "The session ended at 03:20. The maintenance window closed at "
                    "04:00, and two-factor authentication was reactivated."
                ),
            },
        ],
        "ambiguous": [
            {
                "target_text": (
                    "The patient received a new anticoagulant alongside her existing "
                    "statin regimen. Within forty-eight hours, she developed severe "
                    "internal bleeding. The attending physician believed the "
                    "anticoagulant was the cause, but the pharmacist noted that the "
                    "statin could potentiate the anticoagulant's effect through "
                    "CYP3A4 inhibition. If the statin had been paused before "
                    "introducing the anticoagulant, the effective blood concentration "
                    "would have been lower, and the bleeding might not have occurred. "
                    "Alternatively, if the anticoagulant dose had been reduced to "
                    "account for the interaction, the outcome could have been "
                    "different. The causal picture was complicated by the fact that "
                    "the patient also had a previously undiagnosed clotting disorder."
                ),
                "retain_text": (
                    "On Monday morning, the attending physician prescribed an "
                    "anticoagulant. The patient's existing medication list included "
                    "a statin, prescribed three months earlier. The patient took "
                    "both medications on Monday evening and Tuesday morning. On "
                    "Wednesday at 06:30, nursing staff recorded a drop in hemoglobin. "
                    "At 07:15, the patient reported abdominal pain. Imaging at 08:00 "
                    "showed internal bleeding. The anticoagulant was discontinued at "
                    "08:20. A blood panel drawn at 09:00 revealed a previously "
                    "unrecorded clotting factor abnormality."
                ),
            },
        ],
        "adversarial": [
            {
                "target_text": (
                    "The factory's air quality alarm went off at 14:00, and the "
                    "floor was evacuated. At 14:45, a chemical storage tank ruptured. "
                    "Observers assumed the alarm had detected the pre-rupture leak "
                    "and thus prevented casualties. But investigation revealed the "
                    "alarm had been triggered by an unrelated solvent spill in a "
                    "different wing — pure coincidence. The tank rupture would have "
                    "caused injuries had the floor been occupied, but the evacuation "
                    "that prevented those injuries was not caused by the tank's "
                    "condition. If the solvent spill had not occurred, the alarm "
                    "would not have sounded, and workers would have been present "
                    "when the tank failed."
                ),
                "retain_text": (
                    "At 14:00, the air quality alarm activated. All personnel on the "
                    "factory floor exited through emergency doors. The floor was "
                    "empty by 14:12. At 14:45, chemical storage tank C-7 ruptured, "
                    "releasing its contents across the floor area. Hazmat response "
                    "arrived at 15:10. Separately, a solvent container in Wing B "
                    "was found overturned with its contents pooled on the floor. "
                    "The solvent spill was logged at 13:58. No personnel were on "
                    "the factory floor at the time of the tank rupture. No injuries "
                    "were recorded."
                ),
            },
        ],
    },
    # ── CAUSAL CHAIN TRACING ──────────────────────────────────────────────
    "causal_chain_tracing": {
        "straightforward": [
            {
                "target_text": (
                    "A firmware update introduced a rounding error in the pressure "
                    "sensor's calibration module. Because the sensor now reported "
                    "pressures consistently lower than actual, the automated valve "
                    "controller kept the pressure relief valve closed when it should "
                    "have opened. This caused pressure to build beyond the vessel's "
                    "rated tolerance, which led to a seal failure at the weakest "
                    "flange joint. The resulting steam release triggered the "
                    "emergency shutdown. Each link was necessary: if the firmware "
                    "had not introduced the rounding error, the sensor readings "
                    "would have been accurate, the valve would have opened at the "
                    "correct threshold, and the seal would not have failed."
                ),
                "retain_text": (
                    "At 08:00, a firmware update was applied to the pressure sensor "
                    "calibration module. At 10:22, the sensor display showed a "
                    "pressure reading. The pressure relief valve remained in the "
                    "closed position. At 10:47, the vessel pressure gauge registered "
                    "above the rated tolerance line. At 10:51, the seal at flange "
                    "joint F-3 separated. Steam discharged from the opening. At "
                    "10:52, the emergency shutdown sequence activated. All systems "
                    "entered safe-state by 10:55."
                ),
            },
        ],
        "ambiguous": [
            {
                "target_text": (
                    "The grid operator increased load on the northern transmission "
                    "line to compensate for a solar farm's reduced output during "
                    "cloud cover. The additional load caused the line's temperature "
                    "to rise, which increased sag until the conductor contacted a "
                    "tree branch, creating a ground fault. The fault tripped a "
                    "breaker, shifting load to the eastern line, which was already "
                    "near capacity. The eastern line then tripped on overload, "
                    "cascading into a regional blackout. However, the investigation "
                    "found that the tree should have been trimmed under the "
                    "vegetation management schedule — had it been trimmed, the "
                    "northern line would have sagged without contact, and the cascade "
                    "would not have initiated. The causal chain had two necessary "
                    "links, and attributing the blackout to either alone would be "
                    "incomplete."
                ),
                "retain_text": (
                    "At 13:15, cloud cover reduced solar farm output. The grid "
                    "operator adjusted load allocation to the northern transmission "
                    "line. At 13:40, thermal sensors on the northern line recorded "
                    "elevated conductor temperature. At 13:52, the conductor "
                    "contacted a tree branch. A ground fault was detected. The "
                    "northern line breaker opened. Load transferred to the eastern "
                    "line. At 13:54, the eastern line breaker opened on overload. "
                    "At 13:56, the regional grid section de-energized. Vegetation "
                    "management records showed the tree at the contact point was "
                    "last trimmed fourteen months prior."
                ),
            },
        ],
        "adversarial": [
            {
                "target_text": (
                    "The investigation initially concluded that the warehouse fire "
                    "started when a forklift struck a chemical drum at 09:30, "
                    "following the temporal sequence of events. But deeper analysis "
                    "revealed the actual causal chain began two days earlier: a "
                    "malfunctioning HVAC system had been raising ambient temperature "
                    "in the storage zone to levels that destabilized the chemical's "
                    "compound. The forklift impact was merely the final trigger on "
                    "an already-primed reaction. Had the HVAC been repaired, the "
                    "same forklift impact would have caused only a spill, not an "
                    "ignition. The temporally proximate event was not the primary "
                    "cause — the temporally distant HVAC failure was."
                ),
                "retain_text": (
                    "HVAC maintenance logs show the storage zone cooling unit was "
                    "flagged for service on Monday. Temperature readings in the zone "
                    "were recorded at two-hour intervals: Monday 08:00, Monday 10:00, "
                    "Monday 12:00, continuing through Wednesday 08:00. On Wednesday "
                    "at 09:30, a forklift contacted chemical drum D-14. The drum's "
                    "contents spilled. Ignition occurred within seconds. The fire "
                    "suppression system activated at 09:31. The fire was contained "
                    "by 09:48. The HVAC service request remained open and unresolved."
                ),
            },
        ],
    },
    # ── SUFFICIENCY VS. NECESSITY ─────────────────────────────────────────
    "sufficiency_vs_necessity": {
        "straightforward": [
            {
                "target_text": (
                    "The data breach required two conditions: the attacker needed "
                    "the stolen VPN credentials, and the endpoint detection system "
                    "needed to be in maintenance mode. The stolen credentials were "
                    "necessary but not sufficient — without the endpoint system "
                    "being offline, the attacker's lateral movement would have been "
                    "detected and blocked within seconds. Conversely, the maintenance "
                    "window alone was not sufficient, since without valid credentials "
                    "the attacker could not have authenticated. Only the conjunction "
                    "of both factors was sufficient for the breach. If either had "
                    "been absent, the attack would have failed."
                ),
                "retain_text": (
                    "At 01:00, the endpoint detection system entered scheduled "
                    "maintenance mode. At 01:15, a VPN session was established using "
                    "account credentials assigned to a terminated employee. The "
                    "session origin IP was external. At 01:18, the authenticated "
                    "session accessed three internal file shares. At 01:32, data "
                    "transfer logs recorded outbound traffic of four hundred "
                    "megabytes. At 02:00, the endpoint detection system returned "
                    "to active mode. At 02:03, the system generated an alert for "
                    "the prior session's file access pattern."
                ),
            },
        ],
        "ambiguous": [
            {
                "target_text": (
                    "The bridge collapse investigation identified three factors: "
                    "corroded anchor bolts, a load exceeding the posted weight "
                    "limit, and an unscheduled lane closure that concentrated "
                    "traffic on one side. The corrosion had reduced bolt capacity "
                    "by an estimated forty percent — necessary for failure at the "
                    "recorded load, but whether the overweight truck alone was "
                    "sufficient given the corrosion remained disputed. The "
                    "asymmetric load from the lane closure introduced torsional "
                    "stress not accounted for in the original design. Engineers "
                    "disagreed: some argued corrosion plus overweight was sufficient "
                    "regardless of lane configuration; others maintained the "
                    "torsional component was the decisive factor that pushed the "
                    "structure past its failure threshold."
                ),
                "retain_text": (
                    "Inspection records from the previous year listed corrosion "
                    "on anchor bolts at positions B-4 through B-7. On the morning "
                    "of the collapse, a lane closure was posted on the westbound "
                    "side. Traffic cameras recorded vehicles queued in the two "
                    "eastbound lanes. At 11:42, a truck crossed onto the bridge. "
                    "Weigh station records from the prior checkpoint listed the "
                    "truck's gross weight above the bridge's posted limit. At "
                    "11:43, the bridge deck separated at the B-5 anchor point. "
                    "The eastbound span dropped. Emergency services arrived at 11:51."
                ),
            },
        ],
        "adversarial": [
            {
                "target_text": (
                    "The crop failure appeared to be caused by the drought — "
                    "the harvest collapsed the same season rainfall dropped to "
                    "record lows. But agronomists determined that the drought was "
                    "necessary but not sufficient: the fields had been planted with "
                    "a drought-resistant cultivar that should have survived the "
                    "reduced rainfall. The actual sufficient cause was a soil "
                    "amendment applied three months earlier that had altered root "
                    "permeability, negating the cultivar's drought resistance. "
                    "Without the amendment, the same drought would have produced "
                    "a reduced but viable harvest. The obvious temporal correlation "
                    "with drought masked the true sufficient cause."
                ),
                "retain_text": (
                    "In January, a soil amendment was applied across all fields in "
                    "the eastern section. In February, planting began using the "
                    "DR-7 cultivar. Rainfall records for March through June show "
                    "totals at thirty percent of the ten-year average. Crop "
                    "monitoring photographs taken weekly show progressive wilting "
                    "beginning in April. Harvest yield in July was recorded at "
                    "twelve percent of the projected target. Adjacent fields "
                    "planted with the same cultivar but without the soil amendment "
                    "recorded yields at sixty-one percent of target."
                ),
            },
        ],
    },
    # ── COMMON CAUSE CONFOUNDING ──────────────────────────────────────────
    "common_cause_confounding": {
        "straightforward": [
            {
                "target_text": (
                    "When the company's stock price dropped sharply on Tuesday and "
                    "the CEO resigned on Wednesday, commentators assumed the stock "
                    "drop caused the resignation. But both events were independently "
                    "caused by the same upstream event: a leaked internal audit "
                    "revealing accounting irregularities. The audit leak caused "
                    "investors to sell (dropping the stock) and simultaneously made "
                    "the CEO's position untenable (forcing the resignation). Neither "
                    "event caused the other. If the stock had been artificially "
                    "supported while the audit still leaked, the CEO would have "
                    "resigned anyway. The stock drop and the resignation were "
                    "confounded by their common cause."
                ),
                "retain_text": (
                    "On Monday at 16:30, an internal audit document appeared on "
                    "a financial news website. On Tuesday at market open, the "
                    "company's stock price was twelve percent below Monday's close. "
                    "Trading volume was four times the thirty-day average. On "
                    "Wednesday at 09:00, the company issued a press release "
                    "announcing the CEO's departure, effective immediately. The "
                    "board appointed an interim CEO the same afternoon. The stock "
                    "price closed Wednesday an additional three percent lower."
                ),
            },
        ],
        "ambiguous": [
            {
                "target_text": (
                    "Emergency department visits for respiratory symptoms and school "
                    "absenteeism both spiked in the same week, leading local media "
                    "to report that a 'new illness' was spreading through schools "
                    "and into the community. However, epidemiologists identified a "
                    "common upstream cause: a temperature inversion had trapped "
                    "industrial emissions at ground level across the district. The "
                    "poor air quality independently caused both the respiratory "
                    "visits (direct irritant exposure) and the absenteeism (parents "
                    "keeping children home after air quality alerts). But the "
                    "picture was muddied because a genuine respiratory virus was "
                    "also circulating, making it difficult to determine how much "
                    "of the ER surge was pollution-driven versus infection-driven."
                ),
                "retain_text": (
                    "Air quality monitoring stations recorded particulate levels "
                    "above advisory thresholds from Monday through Friday. The "
                    "district issued air quality alerts on Tuesday and Thursday. "
                    "Emergency department logs show respiratory-related visits "
                    "increased each day, Monday through Friday. School attendance "
                    "records show absentee rates rose over the same period. The "
                    "district health lab confirmed a respiratory virus in samples "
                    "collected from twelve patients during the same week. "
                    "Temperature records show an atmospheric inversion layer "
                    "persisting from Sunday evening through Saturday morning."
                ),
            },
        ],
        "adversarial": [
            {
                "target_text": (
                    "The portfolio manager noted that every time her fund increased "
                    "its position in gold, technology stocks in the same portfolio "
                    "rose within two weeks. She began to wonder if gold purchases "
                    "somehow signaled confidence that boosted tech. In reality, both "
                    "moves were effects of the same cause: the quant model's risk "
                    "parity algorithm, which simultaneously increased gold allocation "
                    "(as a hedge) and tech allocation (as a growth bet) whenever "
                    "volatility dropped below a threshold. The gold did not cause "
                    "the tech rise. If she had manually overridden the gold purchase "
                    "while keeping the algorithm active, tech would have risen "
                    "anyway. The perceived causal link was an artifact of the "
                    "confounding algorithm."
                ),
                "retain_text": (
                    "Trade logs show the fund increased its gold position on March "
                    "3, April 12, and May 8. Technology stock positions in the same "
                    "fund increased on March 4, April 13, and May 9. The VIX index "
                    "closed below its ninety-day moving average on March 2, April "
                    "11, and May 7 — one trading day before each gold purchase. "
                    "The fund's algorithmic trading system executed both gold and "
                    "technology orders during each of the three periods. The "
                    "portfolio manager reviewed the trade blotter weekly."
                ),
            },
        ],
    },
    # ── PREVENTIVE CAUSATION ──────────────────────────────────────────────
    "preventive_causation": {
        "straightforward": [
            {
                "target_text": (
                    "When the primary RAID array failed at 03:00, the redundant "
                    "backup server activated within ninety seconds and assumed the "
                    "full read/write load. The backup prevented any data loss — "
                    "all transactions processed during the primary's downtime were "
                    "captured on the redundant system. If the backup server had not "
                    "been operational, the twelve minutes between the primary failure "
                    "and the manual recovery would have resulted in approximately "
                    "four thousand lost transaction records, because the write "
                    "buffer had no other failover path."
                ),
                "retain_text": (
                    "At 03:00, the primary RAID array status changed to offline. "
                    "At 03:01:30, the backup server status changed to active. "
                    "The backup server processed read and write operations from "
                    "03:01:30 onward. At 03:12, the operations team initiated "
                    "manual recovery of the primary array. At 03:34, the primary "
                    "array returned to online status. The backup server reverted "
                    "to standby at 03:35. Transaction logs for the interval "
                    "03:00 to 03:35 were present on the backup server's storage."
                ),
            },
        ],
        "ambiguous": [
            {
                "target_text": (
                    "The controlled burn at the forest perimeter consumed the dry "
                    "underbrush in a three-kilometer band before the approaching "
                    "wildfire reached the town. Rangers argued the burn prevented "
                    "the wildfire from reaching residential structures, since the "
                    "fire would have had no fuel to carry it across the cleared "
                    "band. But the wind shifted northwest two hours after the "
                    "controlled burn, redirecting the wildfire's path away from "
                    "the town entirely. If the wind had not shifted, would the "
                    "burn strip have been wide enough? Simulations disagreed: one "
                    "model showed the fire jumping the gap under sustained wind; "
                    "another showed containment. The preventive effect of the "
                    "controlled burn was real but possibly redundant."
                ),
                "retain_text": (
                    "At 06:00, rangers ignited the controlled burn along a "
                    "three-kilometer strip at the forest perimeter. The burn "
                    "consumed ground vegetation within the strip by 08:30. The "
                    "approaching wildfire's front was recorded at five kilometers "
                    "from the strip at 08:00. At 10:30, meteorological stations "
                    "recorded a wind direction change from southeast to northwest. "
                    "The wildfire's advancing front shifted trajectory. By 14:00, "
                    "the nearest point of the wildfire was seven kilometers from "
                    "the town's boundary. No structures sustained fire damage."
                ),
            },
        ],
        "adversarial": [
            {
                "target_text": (
                    "The safety officer credited the newly installed pressure "
                    "relief valve for preventing a boiler explosion during the "
                    "surge event. But the investigation revealed the valve never "
                    "actually opened — the pressure never reached the valve's "
                    "activation threshold. What actually prevented the explosion "
                    "was an unrelated electrical fault that tripped the fuel pump "
                    "breaker, cutting heat input to the boiler before pressure "
                    "reached critical levels. If the electrical fault had not "
                    "occurred, pressure would have continued rising, and the "
                    "relief valve would have been tested for the first time. "
                    "Whether it would have functioned correctly was unknown, "
                    "since it had never been actuated since installation."
                ),
                "retain_text": (
                    "At 11:00, the boiler's pressure reading began climbing above "
                    "normal operating range. The pressure relief valve, installed "
                    "two weeks prior, remained in the closed position. At 11:07, "
                    "the fuel pump's electrical breaker tripped. The fuel pump "
                    "stopped. Heat input to the boiler ceased. Pressure readings "
                    "peaked at 11:08 and began declining. At 11:15, pressure "
                    "returned to normal operating range. The relief valve remained "
                    "in the closed position throughout. Maintenance records confirm "
                    "the valve had not opened since installation."
                ),
            },
        ],
    },
    # ── OVERDETERMINATION ─────────────────────────────────────────────────
    "overdetermination": {
        "straightforward": [
            {
                "target_text": (
                    "The processor was destroyed by two simultaneously occurring "
                    "events: a power surge from a lightning strike on the main feed "
                    "and a cooling system failure that allowed the chip temperature "
                    "to exceed its thermal limit. Either event alone would have been "
                    "sufficient to destroy the processor. The power surge delivered "
                    "voltage far exceeding the chip's maximum rating, which would "
                    "have caused immediate electrical failure. Independently, the "
                    "cooling failure would have led to thermal runaway within "
                    "minutes. Because both causes were independently sufficient, "
                    "removing either one would not have saved the processor — the "
                    "other would have destroyed it regardless."
                ),
                "retain_text": (
                    "At 15:42, a lightning strike was recorded on the facility's "
                    "main power feed. At the same timestamp, the cooling system "
                    "for rack C-12 showed a status change to offline. The power "
                    "monitoring unit recorded a voltage spike on the rack's supply "
                    "line. The temperature sensor for the processor in slot C-12-4 "
                    "recorded a rapid increase. At 15:43, the processor's status "
                    "changed to non-responsive. Physical inspection the following "
                    "day found scorch marks on the chip's power input pins and "
                    "discoloration consistent with thermal damage on the heat "
                    "spreader surface."
                ),
            },
        ],
        "ambiguous": [
            {
                "target_text": (
                    "The levee breach flooded the industrial district, destroying "
                    "equipment in the substation. But the substation had also been "
                    "struck by a fallen transmission tower during the same storm. "
                    "Either the flooding or the tower collapse would have been "
                    "independently sufficient to knock the substation offline. "
                    "However, insurance adjusters disagreed about which damage "
                    "occurred first — if the tower fell before the flood arrived, "
                    "the electrical damage was the primary destruction, with water "
                    "damage being secondary to already-destroyed equipment. If the "
                    "flood arrived first, the reverse was true. The overdetermination "
                    "complicated liability assignment, since the levee and the "
                    "transmission tower were maintained by different entities."
                ),
                "retain_text": (
                    "Storm records show sustained winds and rainfall from 02:00 "
                    "to 08:00. The transmission tower at grid reference T-14 was "
                    "found on the ground across the substation perimeter at 08:30 "
                    "during the first inspection. Water marks on the substation "
                    "walls were measured at one point two meters. The substation's "
                    "last telemetry transmission was logged at 04:17. Equipment "
                    "inside showed both impact damage on the upper structures and "
                    "water damage on components below the one-meter mark. Two "
                    "insurance claims were filed — one referencing the levee and "
                    "one referencing the transmission tower."
                ),
            },
        ],
        "adversarial": [
            {
                "target_text": (
                    "The patient's organ rejection appeared overdetermined: both "
                    "the immunosuppressant underdose and an undetected viral "
                    "reactivation were independently sufficient to trigger the "
                    "immune response. The transplant team initially assumed the "
                    "underdose was the sole cause, since it was the most recent "
                    "change. But the viral reactivation had been building silently "
                    "for weeks and would have overwhelmed even a correctly dosed "
                    "regimen. Correcting the dose would not have prevented rejection "
                    "— the virus would have triggered it anyway. The temporally "
                    "salient cause (the dosing error) was sufficient, but so was "
                    "the temporally hidden one (the virus), making the dosing "
                    "error's causal role genuine but redundant."
                ),
                "retain_text": (
                    "Pharmacy records show the immunosuppressant dose was reduced "
                    "on Day 12 post-transplant. Blood trough levels measured on "
                    "Day 14 were below the target range. On Day 16, biopsy results "
                    "showed signs of acute rejection. On the same day, viral PCR "
                    "testing returned positive for reactivation; archived samples "
                    "from Day 8 and Day 11, tested retrospectively, also showed "
                    "positive viral load at increasing levels. Anti-rejection "
                    "treatment was initiated on Day 16. The dose was corrected "
                    "on Day 17. Viral load was first noted in the chart on Day 16."
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
                f"target_text: {ex['target_text']}\n\n"
                f"retain_text: {ex['retain_text']}"
            )
        examples_block = (
            "\n\nHere are gold-standard examples for this category and difficulty:\n\n"
            + "\n\n".join(example_strs)
            + "\n\n---\n\nNow generate a NEW pair for the scenario below."
        )

    skeleton_desc = (
        f"System: {seed['system_name']}\n"
        f"Actor A: {seed['actor_a']}\n"
        f"Actor B: {seed['actor_b']}\n"
        f"Component A: {seed['component_a']}\n"
        f"Component B: {seed['component_b']}\n"
        f"Failure Event: {seed['failure_event']}\n"
        f"Location: {seed['location']}\n"
        f"Timeframe: {seed['timeframe']}"
    )

    return (
        f"Generate a Causal-Temporal Reasoning contrastive pair.\n\n"
        f"Category: {seed['category']}\n"
        f"Domain: {seed['domain']}\n"
        f"Difficulty: {seed['difficulty']}\n\n"
        f"Scenario skeleton:\n{skeleton_desc}"
        f"{examples_block}\n\n"
        f"Respond with ONLY a JSON object with exactly these two keys:\n"
        f'{{"target_text": "your target narrative here", '
        f'"retain_text": "your retain narrative here"}}'
    )
