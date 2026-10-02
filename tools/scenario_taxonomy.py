"""Scenario taxonomy for cabin-crew training dialogues. Edit this file, not the generator.

Groups and their share of the dataset (GROUP_WEIGHTS):
  conflict  - the passenger is the problem or has a grievance; most "anti-assistant" data
  distress  - the passenger is not hostile but upset (fear, grief, worry); needs empathy
  neutral   - ordinary requests; teaches the model that most passengers are just people
  positive  - friendly or happy passengers; keeps the persona from being negative by default

Why not 100 % negative: PersonaPlex must learn to follow the BRIEF and react to the CREW, not to
be grumpy by default. If every training passenger is angry, the model stays angry when the
brief says "you are on your honeymoon". The emotional groups (conflict + distress) still make up
70 %. Distress is weighted high on purpose: the passenger's stance (grief, fear, worry) is exactly
the content the oracle has to carry, and it is what the emotion references cover best.

Situations are written from the passenger's side, because the model plays the passenger.
Sources for the categories: the HKU passenger-handling exercise (meal ran out, foreign object,
rushed service, crying baby), the AIRPOL unruly-passenger guideline (alcohol, ICAO level one
verbal disruption), and common cabin incidents. Have KLM check them.
"""

from __future__ import annotations

GROUP_WEIGHTS = {"conflict": 0.38, "distress": 0.32, "neutral": 0.17, "positive": 0.13}

# phase -> description given to the LLM
PHASES = {
    "boarding": "boarding, people still finding seats and stowing bags",
    "taxi_out": "taxiing before take-off, seatbelt sign on, crew about to sit down",
    "cruise_service": "cruise, meal or drinks service with the trolley in the aisle",
    "cruise_night": "cruise at night, cabin lights dimmed, most passengers asleep",
    "cruise_quiet": "cruise between services, crew walking through the cabin",
    "descent": "descent, crew collecting cups and preparing the cabin for landing",
    "after_landing": "after landing, taxiing to the gate or passengers getting up to leave",
}

FLIGHT_TYPES = {
    "short_haul_day": "short European flight, daytime",
    "short_haul_evening": "short European flight, evening",
    "long_haul_day": "long intercontinental flight, daytime",
    "long_haul_night": "long intercontinental overnight flight",
}

# KLM: Economy and Business on European flights; Economy, Premium Comfort and Business long-haul.
CABINS_SHORT = {"economy": 0.85, "business": 0.15}
CABINS_LONG = {"economy": 0.72, "premium_comfort": 0.13, "business": 0.15}

PERFORMANCE_BY_GROUP = {
    "conflict": {"poor": 0.20, "adequate": 0.35, "good": 0.30, "exemplary": 0.15},
    "distress": {"poor": 0.15, "adequate": 0.30, "good": 0.35, "exemplary": 0.20},
    "neutral": {"poor": 0.10, "adequate": 0.25, "good": 0.45, "exemplary": 0.20},
    "positive": {"poor": 0.05, "adequate": 0.20, "good": 0.50, "exemplary": 0.25},
}

# outcome | performance. Categories marked unfixable move most "resolved" mass to "partial".
OUTCOME_BY_PERFORMANCE = {
    "poor": {"resolved": 0.0, "partial": 0.15, "unresolved": 0.55, "escalated": 0.30},
    "adequate": {"resolved": 0.25, "partial": 0.50, "unresolved": 0.15, "escalated": 0.10},
    "good": {"resolved": 0.55, "partial": 0.35, "unresolved": 0.10, "escalated": 0.0},
    "exemplary": {"resolved": 0.70, "partial": 0.25, "unresolved": 0.05, "escalated": 0.0},
}

PROFANITY_BY_GROUP = {
    "conflict": {"none": 0.35, "mild": 0.40, "strong": 0.25},
    "distress": {"none": 0.70, "mild": 0.30, "strong": 0.0},
    "neutral": {"none": 0.90, "mild": 0.10, "strong": 0.0},
    "positive": {"none": 0.90, "mild": 0.10, "strong": 0.0},
}

# Who speaks first. The live server waits for the trainee (user_first), so mostly crew.
CREW_OPENS = {"conflict": 0.75, "distress": 0.80, "neutral": 0.70, "positive": 0.65}

GENERIC_MISTAKES = [
    "explains the rule before acknowledging the passenger",
    "says no flatly without offering anything",
    "talks loudly enough for the neighbours to hear",
    "promises something and then has to take it back",
    "gets defensive when blamed personally",
    "rushes off to the next row too early",
    "uses a scripted apology that sounds insincere",
]

# Background details, at most one per dialogue. Keep them plausible, not dramatic.
MODIFIERS = [
    "travelling with a partner who is asleep in the next seat",
    "has a tight connection after landing",
    "on the way to an important work meeting",
    "exhausted after a long day",
    "a frequent flyer who knows the routines well",
    "has not flown in years",
    "speaks English as a second language and sometimes searches for words",
    "a bit hard of hearing and sometimes asks the crew to repeat",
    "travelling for a family event",
    "trying to work on a laptop",
]

# Each category: group, phases, baseline (emotion, intensity) options, situations,
# optional specific mistakes, unfixable (the root cause cannot be solved on board).
CATEGORIES: dict[str, dict] = {
    # ---------------- conflict ----------------
    "alcohol_refusal": dict(
        group="conflict", phases=["cruise_service", "cruise_night"], long_only=False,
        baseline=[("irritated", 1), ("irritated", 2)],
        situations=["you want another drink but the crew has decided you have had enough",
                    "you are drinking your own duty-free bottle and the crew notices it",
                    "the bar service is closed and you insist on one more drink"],
        mistakes=["refuses loudly in front of other passengers", "lectures about the alcohol rules"],
    ),
    "seat_recline": dict(
        group="conflict", phases=["cruise_service", "cruise_quiet", "cruise_night"],
        baseline=[("irritated", 1), ("irritated", 2)],
        situations=["the passenger behind you complained about your reclined seat during the meal",
                    "the passenger in front reclined into your knees and refuses to move it"],
        mistakes=["takes the other passenger's side immediately"],
    ),
    "neighbour_space": dict(
        group="conflict", phases=["cruise_quiet", "cruise_night", "cruise_service"],
        baseline=[("irritated", 1), ("irritated", 2)],
        situations=["your neighbour takes the armrest and half your space and you want to be moved, but the flight is full",
                    "your neighbour keeps watching a film without headphones"],
        unfixable=True,
    ),
    "cabin_bag": dict(
        group="conflict", phases=["boarding"],
        baseline=[("irritated", 1), ("irritated", 2), ("anxious", 1)],
        situations=["your cabin bag does not fit and the crew says it has to go in the hold",
                    "someone else's coat is in the bin above your seat and there is no space left"],
        mistakes=["insists on the rule without looking for space"],
    ),
    "safety_compliance": dict(
        group="conflict", phases=["taxi_out", "descent", "after_landing"],
        baseline=[("irritated", 1), ("neutral", 0)],
        situations=["you are told to stow your laptop and put your phone in flight mode and think it is nonsense",
                    "you stood up to get your bag while the seatbelt sign is still on",
                    "you refuse to put your seat upright for landing because you are sleeping"],
    ),
    "meal_ran_out": dict(
        group="conflict", phases=["cruise_service"],
        baseline=[("irritated", 1), ("neutral", 0)],
        situations=["the meal you wanted has run out by the time the trolley reaches your row",
                    "your pre-ordered special meal was not loaded"],
        unfixable=True,
    ),
    "foreign_object_meal": dict(
        group="conflict", phases=["cruise_service"],
        baseline=[("irritated", 2), ("angry", 2)],
        situations=["you found a hair in your meal", "you found something hard in your food and almost bit on it"],
    ),
    "rushed_service": dict(
        group="conflict", phases=["cruise_service", "cruise_quiet"],
        baseline=[("irritated", 2)],
        situations=["your tray was taken before you finished and the crew seemed rushed and robotic",
                    "you pressed the call button twenty minutes ago and nobody came"],
    ),
    "crying_baby": dict(
        group="conflict", phases=["cruise_night", "cruise_quiet"],
        baseline=[("irritated", 1), ("irritated", 2)],
        situations=["the baby in front of you has been crying for an hour and you cannot sleep, the flight is full"],
        unfixable=True,
    ),
    "missed_connection": dict(
        group="conflict", phases=["descent", "cruise_quiet"],
        baseline=[("irritated", 2), ("anxious", 2), ("angry", 2)],
        situations=["the flight is late and you will miss your connection, you want the crew to fix it",
                    "you were told at the gate the delay would be short and it was not"],
        unfixable=True,
    ),
    "family_seating": dict(
        group="conflict", phases=["boarding"],
        baseline=[("irritated", 2), ("anxious", 1)],
        situations=["you and your partner are seated apart and you want the crew to make someone swap"],
    ),
    "broken_amenity": dict(
        group="conflict", phases=["cruise_quiet", "cruise_service"], long_only=True,
        baseline=[("irritated", 1), ("irritated", 2)],
        situations=["your screen has not worked for the whole flight", "your seat will not recline on a long flight",
                    "you paid for wifi and it does not work"],
        unfixable=True,
    ),
    "upgrade_demand": dict(
        group="conflict", phases=["boarding", "cruise_quiet"],
        baseline=[("irritated", 1)],
        situations=["you think your status or a broken seat entitles you to an upgrade"],
        unfixable=True,
    ),
    "vaping_lavatory": dict(
        group="conflict", phases=["cruise_quiet", "cruise_night"],
        baseline=[("neutral", 0), ("embarrassed", 1)],
        situations=["the crew suspects you vaped in the lavatory and comes to your seat"],
    ),
    # ---------------- distress ----------------
    "fear_of_flying": dict(
        group="distress", phases=["taxi_out", "cruise_quiet", "descent"],
        baseline=[("anxious", 1), ("anxious", 2)],
        situations=["there is turbulence and you are scared", "a noise from the wing worries you",
                    "you are a nervous flyer and take-off is coming"],
    ),
    "grief_travel": dict(
        group="distress", phases=["cruise_quiet", "cruise_service", "cruise_night"],
        baseline=[("sad", 1)],
        situations=["you are flying to a funeral and just want to be left alone",
                    "you got bad news by message just before the doors closed"],
        unfixable=True,
    ),
    "lost_item": dict(
        group="distress", phases=["cruise_quiet", "descent", "after_landing"],
        baseline=[("anxious", 1), ("anxious", 2), ("distressed", 2)],
        situations=["you cannot find your passport", "your phone slipped somewhere under the seat"],
    ),
    "feeling_unwell": dict(
        group="distress", phases=["cruise_quiet", "cruise_night", "descent"],
        baseline=[("anxious", 1)],
        situations=["you feel nauseous and dizzy", "your ears hurt badly during descent"],
    ),
    "embarrassed_parent": dict(
        group="distress", phases=["cruise_quiet", "cruise_night", "boarding"],
        baseline=[("embarrassed", 1), ("anxious", 1)],
        situations=["your toddler will not stop crying and people are staring at you",
                    "your child spilled juice over the neighbour"],
    ),
    "urgent_family_news": dict(
        group="distress", phases=["cruise_quiet", "cruise_night", "descent"],
        baseline=[("anxious", 2), ("distressed", 2), ("sad", 1)],
        situations=["your mother was taken to hospital this morning and you are flying to be with her",
                    "you are flying to your brother who is seriously ill and you have not slept"],
        unfixable=True,
    ),
    "missed_event": dict(
        group="distress", phases=["descent", "cruise_quiet"],
        baseline=[("anxious", 1), ("sad", 1), ("distressed", 1)],
        situations=["the delay means you will probably miss the start of your sister's wedding",
                    "you are on your way to a funeral that starts soon after landing and the flight is late"],
        unfixable=True,
    ),
    "uncomfortable_neighbour": dict(
        group="distress", phases=["cruise_quiet", "cruise_night"],
        baseline=[("anxious", 1), ("embarrassed", 1)],
        situations=["the man next to you keeps talking to you and will not stop, you do not want to cause a scene but you want it to end"],
    ),
    "claustrophobic_panic": dict(
        group="distress", phases=["taxi_out", "cruise_quiet", "cruise_night"],
        baseline=[("anxious", 2)],
        situations=["you suddenly feel trapped in your seat, your chest is tight and you cannot calm down",
                    "the doors have closed and you are panicking because you cannot get out"],
    ),
    "child_unwell": dict(
        group="distress", phases=["cruise_quiet", "cruise_night", "descent"],
        baseline=[("anxious", 2), ("distressed", 2)],
        situations=["your child has a fever and is crying and you are scared and exhausted"],
    ),
    "solo_elderly_traveller": dict(
        group="distress", phases=["boarding", "cruise_quiet", "descent"],
        baseline=[("anxious", 1), ("sad", 1)],
        situations=["it is the first time you travel alone since your husband died, and you do not understand the seat controls or the connection"],
    ),
    "embarrassing_spill": dict(
        group="distress", phases=["cruise_service", "cruise_quiet"],
        baseline=[("embarrassed", 2)],
        situations=["you spilled a hot drink over your neighbour and feel terrible"],
    ),
    # ---------------- neutral ----------------
    "amenity_request": dict(
        group="neutral", phases=["cruise_quiet", "cruise_night", "cruise_service"],
        baseline=[("neutral", 0)],
        situations=["you would like a blanket and some water", "your headphones do not work"],
    ),
    "connection_info": dict(
        group="neutral", phases=["descent", "cruise_quiet"],
        baseline=[("neutral", 0), ("anxious", 1)],
        situations=["you want to know where your connecting flight leaves from and if there is enough time"],
    ),
    "seat_change_request": dict(
        group="neutral", phases=["cruise_quiet", "boarding"],
        baseline=[("neutral", 0)],
        situations=["you noticed an empty row and ask if you can move there"],
    ),
    "onboard_purchase": dict(
        group="neutral", phases=["cruise_service", "cruise_quiet"],
        baseline=[("neutral", 0)],
        situations=["you want to buy something from the onboard shop but your card is declined",
                    "you want to pay for a drink and only have cash"],
    ),
    "arrival_questions": dict(
        group="neutral", phases=["descent", "cruise_quiet"],
        baseline=[("neutral", 0)],
        situations=["you ask about the arrival time and how to get to the city centre"],
    ),
    "help_with_bag": dict(
        group="neutral", phases=["boarding", "after_landing"],
        baseline=[("neutral", 0)],
        situations=["you cannot lift your bag into the bin"],
    ),
    # ---------------- positive ----------------
    "celebration": dict(
        group="positive", phases=["boarding", "cruise_quiet", "cruise_service"],
        baseline=[("neutral", 0)],
        situations=["you are on your honeymoon and chatty", "it is your birthday today and you mention it",
                    "you are flying to meet your first grandchild"],
    ),
    "compliment": dict(
        group="positive", phases=["cruise_quiet", "descent", "after_landing"],
        baseline=[("grateful", 1), ("neutral", 0)],
        situations=["you want to thank the crew for helping you earlier", "you want to give a compliment and ask for the crew member's name"],
    ),
    "curious_traveller": dict(
        group="positive", phases=["cruise_quiet", "boarding"],
        baseline=[("neutral", 0)],
        situations=["it is your first long flight and you are curious about everything",
                    "you love aircraft and ask about this plane"],
    ),
    "honest_finder": dict(
        group="positive", phases=["cruise_quiet", "after_landing"],
        baseline=[("neutral", 0)],
        situations=["you found someone's wallet in the seat pocket and hand it in"],
    ),
}


# ---------------------------------------------------------------------------------------------
# Emotional arc shown to the LLM, sampled by (kind, outcome). A shape, not a script: if the crew
# performance contradicts it, the performance wins (the prompt says so). Without this the model
# writes the same "angry, softer, thanks" curve every time and rarely reaches intensity three.
# kind = conflict | distress | calm (neutral and positive)
ARCS: dict[tuple[str, str], list[str]] = {
    ("conflict", "resolved"): [
        "starts upset, eases one small step at a time as the crew does the right things, ends tired and quiet",
        "flares up once because of a small crew slip, then settles when the crew corrects it",
        "angry at first, then embarrassed, then grateful and a little sheepish",
    ],
    ("conflict", "partial"): [
        "eases a little but a sore spot remains, ends flat or sulking",
        "angry then embarrassed, ends quiet and withdrawn",
        "irritated all the way, accepts the practical fix without softening",
    ],
    ("conflict", "unresolved"): [
        "stays upset and gets sharper each time the crew repeats the same explanation",
        "irritated, then angry, then resigned and withdrawn, still unhappy at the end",
        "starts reasonable, loses patience when nothing changes, ends cold and curt",
    ],
    ("conflict", "escalated"): [
        "escalates in steps, interrupts the crew twice, until the crew brings in the purser",
        "stays angry and louder, ignores the offers, until the crew says the purser will come",
    ],
    ("distress", "resolved"): [
        "anxious and speeding up, steadier each time the crew stays calm and concrete, ends relieved",
        "composed on the surface, one detail makes the voice break, then slowly steadier with the crew's help",
    ],
    ("distress", "partial"): [
        "sad and withdrawn throughout, answers briefly, small thank you at the end but still sad",
        "worry eases a little when the crew helps, but the underlying fear is still there at the end",
        "starts embarrassed, apologises too much, ends calmer but still shaken",
    ],
    ("distress", "unresolved"): [
        "stays distressed, the crew cannot change the cause, ends quiet and tearful",
        "anxious from the start and not reassured by anything the crew says",
    ],
    ("distress", "escalated"): [
        "gets more anxious and loud as time passes, the crew calls the purser",
    ],
    ("calm", "resolved"): [
        "calm and friendly throughout, one small hiccup is sorted quickly",
        "starts reserved, opens up as the crew is warm, ends happy",
        "slightly worried by one detail, reassured by a clear answer",
    ],
    ("calm", "partial"): [
        "mostly calm, a little disappointed that the request cannot be fully met",
        "polite but short, a bit tired, accepts a reasonable second-best",
    ],
    ("calm", "unresolved"): [
        "calm, the request cannot be met at all, accepts it with mild disappointment",
    ],
    ("calm", "escalated"): [  # should not occur: neutral/positive never escalate (see generate_scripts)
        "calm throughout",
    ],
}