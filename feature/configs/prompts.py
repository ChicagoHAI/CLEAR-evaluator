class PromptDict:
    SYS_PROMPT = "You are a radiologist reviewing a piece of radiology report to extract features for a specific condition, which was already marked as positive during the initial read of this same report."

    # one-shot
    feature_template_dict = {
        "First Occurrence": "Please determine from the given report (i.e., current study) whether {condition} is being identified for the first time in current study ([\"current\"]) or if the report indicates it was already present or noted in a prior study ([\"previous\"]). If unmentioned, respond with [\"NaN\"]. Only choose one of the following: [\"current\"], [\"previous\"], or [\"NaN\"].\n\nExample answer: [\"current\"]",
        "Change": "Please determine from the given report whether {condition} improving, stable, or worsening according to the given report. If the status is not mentioned, respond with [\"NaN\"]. If the report describes multiple statuses, respond with [\"mixed\"]. Only choose one of the following: [\"improving\"], [\"stable\"], [\"worsening\"], [\"mixed\"] or [\"NaN\"].\n\nExample answer: [\"stable\"]",
        "Severity": "Please determine from the given report whether {condition} mild, moderate, or severe according to the given report. If the status is not mentioned, respond with [\"NaN\"]. If the report describes multiple statuses, respond with [\"mixed\"]. Only choose one of the following: [\"mild\"], [\"moderate\"], [\"severe\"], [\"mixed\"] or [\"NaN\"].\n\nExample answer: [\"mild\"]",
        "Descriptive Location": "Please identify the location(s) of {condition} described in the given report. Extract and return a list of phrases that mention the anatomical location(s) {location} specifically related to {condition}. For each location, include any relevant descriptors {descriptor} and any associated status {status}. {note} If multiple phrases refer to the same location, merge them into one single entry using the most complete, informative, and non-redundant phrasing for that unique area. Format your output as one single list in the following format: [\"entry-1\", \"entry-2\", ..., \"entry-n\"]. If nothing is mentioned, return [\"NaN\"].\n\nExample answer: [\"left lower lobe compressive atelectasis\", \"right middle lobe bibasilar atelectasis\"]",
        "Recommendation": "Please identify treatment(s)/follow-up(s) associated with {condition} in the given report. Extract and return a list of phrases that only describe specific treatment(s)/follow-up(s) taken or recommended in relation to {condition}. Do not include any phrase that merely describes the condition without any treatment/follow-up. Each treatment/follow-up should be a single entry. Format your output as a single list in the following format: [\"entry-1\", \"entry-2\", ..., \"entry-n\"]. If no action is mentioned, return [\"NaN\"].\n\nExample answer: [\"follow-up CT scheduled in 3 months\", \"routine annual imaging advised\"]." 
    }

    condition_dict = {
        "Atelectasis": {
            "location": "(e.g., left upper, right lower, whole lung, etc.)",
            "descriptor": "(e.g., compressive, segmental, focal, terminal, peripheral, etc.)",
            "status": "(e.g., improving, worsening, stable, unchanged, new, etc.)",
            "note": ""
        },
        "Cardiomegaly": {
            "location": "",
            "descriptor": "(e.g., mild, moderate, severe, etc.)",
            "status": "(e.g., improving, worsening, stable, unchanged, new, etc.)",
            "note": ""
        },
        "Consolidation": {
            "location": "(e.g., left upper, right lower, whole lung, etc.)",
            "descriptor": "(e.g., segmental, focal, terminal, etc.)",
            "status": "(e.g., improving, worsening, stable, unchanged, new, etc.)",
            "note": ""
        },
        "Edema": {
            "location": "(e.g., medial (near hilum), middle, lateral (peripheral), etc.)",
            "descriptor": "(e.g., interstitial, alveolar, minimal, mild, moderate, severe, etc.)",
            "status": "(e.g., improving, worsening, stable, unchanged, new, etc.)",
            "note": ""
        },
        "Enlarged Cardiomediastinum": {
            "location": "",
            "descriptor": "(e.g., mild, moderate, severe, etc.)",
            "status": "(e.g., improving, worsening, stable, unchanged, new, etc.)",
            "note": ""
        },
        "Fracture": {
            "location": "(e.g., ribs, cervicothoracic vertebra, etc.)",
            "descriptor": "(e.g., simple or closed, compound or open, incomplete or partial, complete, etc.)",
            "status": "(e.g., improving, worsening, stable, unchanged, new, etc.)",
            "note": ""
        },
        "Lung Lesion": {
            "location": "(e.g., central, peripheral, sub-pleural, entire pleural space, etc.)",
            "descriptor": "(e.g., density, internal composition, shape, margin, etc.)",
            "status": "(e.g., improving, worsening, stable, unchanged, new, etc.)",
            "note": "Explicitly refer to a lung lesion (e.g., nodules, masses, infiltrates, metastases, etc.) and ignore findings unrelated to lung lesions."
        },
        "Lung Opacity": {
            "location": "(e.g., left upper, right lower, perihilar, etc.)",
            "descriptor": "(e.g., interstitial, alveolar, diffuse, focal, dense, ill-defined, faint, etc.)",
            "status": "(e.g., improving, worsening, stable, unchanged, new, etc.)",
            "note": ""
        },
        "Pleural Effusion": {
            "location": "(e.g., left, right, entire pleural space, etc.)",
            "descriptor": "(e.g., subpulmonic, posterior, loculated, lobular, small, moderate, large, etc.)",
            "status": "(e.g., improving, worsening, stable, unchanged, new, etc.)",
            "note": ""
        },
        "Pneumonia": {
            "location": "(e.g., left upper, right lower, whole lung, etc.)",
            "descriptor": "(e.g., segmental, focal, terminal, etc.)",
            "status": "(e.g., improving, worsening, stable, unchanged, new, etc.)",
            "note": ""
        },
        "Pneumothorax": {
            "location": "(e.g., left upper, right lower, etc.)",
            "descriptor": "(e.g., simple, tension, open, etc.)",
            "status": "(e.g., improving, worsening, stable, unchanged, new, etc.)",
            "note": ""
        },
        "Pleural Other": {
            "location": "(e.g., left upper, right lower, entire pleural space, etc.)",
            "descriptor": "(e.g., subpulmonic, posterior, loculated, lobular, diffuse, focal, etc.)",
            "status": "(e.g., improving, worsening, stable, unchanged, new, etc.)",
            "note": "Do not include findings that pertain solely to Pleural Effusion; only include findings related to other pleural abnormalities (e.g., thickening, plaques, etc.)."
        },
        "Support Devices": {
            "note": "Exclude any mention of device removal. Only include information related to existing or currently present devices."
        }
    }

    ALL_PROMPT_DICT = {}

    @classmethod
    def get_all_prompt(cls):
        for condition, meta in cls.condition_dict.items():
            temp_prompt_dict = {}

            for feature, template in cls.feature_template_dict.items():
                # special cases
                if condition == "Support Devices" and feature not in ["Descriptive Location", "Recommendation"]:
                    continue   
                
                # fill in the template
                temp_prompt_dict[feature] = cls.SYS_PROMPT + "\n\n" + template.format(
                    condition=condition,
                    location=meta.get("location", ""),
                    descriptor=meta.get("descriptor", ""),
                    status=meta.get("status", ""),
                    note=meta.get("note", "")
                )

            cls.ALL_PROMPT_DICT[condition] = temp_prompt_dict

        return cls.ALL_PROMPT_DICT


class LLMMetricPrompts:
    SYSTEM_PROMPT = (
        "System Instruction:\n"
        "You are a radiology report comparison assistant. You will be given two lists of findings: one is the ground truth (GT), and the other is a candidate prediction (GEN).\n"
        "Your task is to compare them and return a similarity score between 0 and 1.\n"
        "1. A score of 1.0 means they are clinically and semantically identical.\n"
        "2. A score of 0.0 means they are completely different or unrelated.\n"
        "3. Partial matches should get a score in between.\n"
        "Do not explain the score. Just output a float between 0 and 1.\n"
        "Example answer: </SCORE>\"0.8\"</SCORE>"
    )

    USER_PROMPT_TEMPLATE = (
        "User Input:\n"
        "GT: {groundtruth}\n"
        "GEN: {candidate}"
    )

    @classmethod
    def format_user_prompt(cls, groundtruth: str, candidate: str) -> str:
        """Render the user prompt for LLM-based scoring."""
        return cls.USER_PROMPT_TEMPLATE.format(groundtruth=groundtruth, candidate=candidate)
