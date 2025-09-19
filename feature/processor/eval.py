# numpy version must be lower than 2.0.0 to adapt nltk
import json
import re
import pandas as pd
import numpy as np
import math
import os
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from openai import AzureOpenAI
from tqdm import tqdm

# Feature configuration
FEATURE_CONFIG = {
    "IE": ['Descriptive Location', "Action"],
    "QA": ['First Occurrence', 'Change', 'Severity', 'Urgency'],
    "Support Devices": ['Descriptive Location', 'Urgency', 'Action'],
    "All": ['First Occurrence', 'Change', 'Severity', 'Descriptive Location', 'Urgency', 'Action']
}

MODEL_CONFIGS = {
    "gpt-4o": {
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "api_version": "2024-02-01",
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "deployment": "gpt-4o",
        "max_tokens": 1024
    },
    "o1-mini": {
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "api_version": "2024-02-01",
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "deployment": "o1-mini",
        "max_tokens": 1024
    }
}


def extract_and_parse_json(text):
    """
    Extract and parse JSON arrays from text or provide reasonable defaults
    
    This function tries multiple approaches to parse JSON from potentially malformed strings:
    1. Direct parsing
    2. Extracting bracket-enclosed content
    3. Looking for conclusion sections
    4. Handling special values
    5. Extracting quoted items
    """
    # 1. Try parsing the entire text directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 2. Try to find and extract JSON arrays
    try:
        # Look for the last bracket-enclosed content
        array_pattern = r'\[(.*?)\](?=[^[\]]*$)'
        match = re.search(array_pattern, text, re.DOTALL)
        if match:
            content = match.group(0)
            # Replace single quotes with double quotes
            content = content.replace("'", '"')
            return json.loads(content)
    except:
        pass
    
    # 3. Try to extract final answer list from longer LLM output
    try:
        # Look for common conclusion markers
        conclusion_markers = ["Therefore", "In conclusion", "So", "Finally", "Hence"]
        for marker in conclusion_markers:
            if marker in text:
                conclusion_part = text.split(marker)[-1].strip()
                # Find bracket content
                array_match = re.search(r'\[(.*?)\]', conclusion_part)
                if array_match:
                    content = array_match.group(0)
                    # Process content to make it valid JSON
                    content = content.replace("'", '"')
                    if '"' not in content:  # Add quotes if missing
                        content = content.replace("[", '["').replace("]", '"]').replace(", ", '", "')
                    return json.loads(content)
    except:
        pass
    
    # 4. Handle special values like "nan" or "NaN"
    if text.strip().lower() in ["nan", "\"nan\"", "'nan'"]:
        return ["nan"]
    
    # 5. Try to extract potential list items from the text
    try:
        # Find all quoted content
        items = re.findall(r'["\'](.*?)["\']', text)
        if items:
            return items
    except:
        pass
    
    # 6. If all attempts fail, return a list containing the original text or default value
    print(f"Unable to parse JSON: {text[:50]}...")
    if text:
        # If text is not empty, return the last sentence or first 50 characters as an element
        last_sentence = text.split(".")[-2].strip() if "." in text else text[:50].strip()
        return [last_sentence]
    else:
        # If text is empty, return default value
        return ["unclear"]
    


def convert_feature_df(dict_gt: dict, 
                       dict_gen: dict, 
                       name: str,
                       mode: str) -> pd.DataFrame:
    """
    Convert feature dictionaries to DataFrame format for evaluation
    """
    temp_data = []
     
    for id in dict_gt:
        assert len(dict_gt[id]) == len(dict_gen[id]), f"{id} has mismatch conditions between gt and gen!"
        gt_condition = dict_gt[id]
        gen_condition = dict_gen[id] 

        for condition in gt_condition:
            if condition == 'Support Devices' and name not in FEATURE_CONFIG["Support Devices"]:
                continue

            gt_answer = gt_condition[condition][name]
            gen_answer = gen_condition[condition][name]

            # # load feature into list
            # try:
            #     gt_feature = json.loads(gt_answer)
            # except json.JSONDecodeError:
            #     gt_feature = json.loads(gt_answer.replace("[", '["').replace("]", '"]'))
            # try:
            #     gen_feature = json.loads(gen_answer)
            # except json.JSONDecodeError:
            #     gen_feature = json.loads(gen_answer.replace("[", '["').replace("]", '"]')) # fix case like: [current]

            # Use the improved JSON extraction function
            gt_feature = extract_and_parse_json(gt_answer)
            gen_feature = extract_and_parse_json(gen_answer)
                
            assert isinstance(gt_feature, list), f"gt {id} {condition} {name} exists incompatible format"
            assert isinstance(gen_feature, list), f"gen {id} {condition} {name} exists incompatible format"

            if mode == 'QA':
                assert len(gt_feature) == 1, f"gt {id} {condition} {name} exists one more answer"
                assert len(gen_feature) == 1, f"gen {id} {condition} {name} exists one more answer"
                # list to str
                gt_content = gt_feature[0].lower()
                gen_content = gen_feature[0].lower()
            elif mode == 'IE':
                gt_content = [x.lower() for x in gt_feature]
                gen_content = [x.lower() for x in gen_feature]             

            temp_data.append({
                "study_id": id,
                "condition": condition,
                "gt_feature": gt_content,
                "gen_feature": gen_content
            })
    
    df_feature = pd.DataFrame(temp_data)
    return df_feature


def get_one_response(usr_prompt, sys_prompt, model_name="o1-mini"):
    """
    Get a response from Azure OpenAI API
    
    Args:
        usr_prompt: User prompt text
        sys_prompt: System prompt text
        model_name: Name of the model to use (must be defined in MODEL_CONFIGS)
        
    Returns:
        Response text from the model
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not found in MODEL_CONFIGS.")
    
    config = MODEL_CONFIGS[model_name]
    
    client = AzureOpenAI(
        api_key=config["api_key"],  
        api_version=config["api_version"],
        base_url=f"{config['endpoint']}/openai/deployments/{config['deployment']}"
    )

    try:
        apiresponse = client.chat.completions.with_raw_response.create(
            model=config["deployment"],
            messages=[
                {
                    "role": "system", 
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": usr_prompt
                }
            ],
            max_tokens=config["max_tokens"]
        )
    except:
        apiresponse = client.chat.completions.with_raw_response.create(
            model=config["deployment"],
            messages=[
                {
                    "role": "user",
                    "content": sys_prompt + "\n\n" + usr_prompt
                }
            ]
        )

    chat_completion = apiresponse.parse()
    response = chat_completion.choices[0].message.content
    
    return response



def compare_model_outputs(gt_path, model_paths, output_dir, feature_types=None, model_names=None, prompt_paths=None, model_name="o1-mini"):
    """
    Compare different model outputs against ground truth using AI scoring
    
    Args:
        gt_path: Path to ground truth JSON file
        model_paths: List of paths to model output JSON files
        output_dir: Directory for output files
        feature_types: Feature types to evaluate, defaults to IE
        model_names: List of model names for output labeling
        prompt_paths: Dictionary with paths to prompt templates
        model_name: Name of model to use for scoring
    """
    print("Comparing model outputs...")
    
    # Handle defaults
    if feature_types is None:
        feature_types = ["IE"]
    if model_names is None:
        model_names = [f"model_{i+1}" for i in range(len(model_paths))]
    if prompt_paths is None:
        prompt_paths = {
            "user": "./usr_prompt.txt",
            "system": "./sys_prompt.txt"
        }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth data
    print(f"Loading ground truth from: {gt_path}")
    with open(gt_path, 'r') as f:
        gt_dict = json.load(f)
    gt_dict = dict(sorted(gt_dict.items()))
    
    # Load model outputs
    model_dicts = []
    for i, path in enumerate(model_paths):
        print(f"Loading {model_names[i]} output from: {path}")
        with open(path, 'r') as f:
            model_dict = json.load(f)
        model_dict = dict(sorted(model_dict.items()))
        # Verify key matching
        assert list(gt_dict.keys()) == list(model_dict.keys()), f"{model_names[i]} keys don't match ground truth."
        model_dicts.append(model_dict)
    
    # Load prompt templates
    with open(prompt_paths["user"], 'r') as f:
        usr_prompt_template = f.read()
    with open(prompt_paths["system"], 'r') as f:
        sys_prompt = f.read()
    
    # Evaluate each feature type
    for feature_type in feature_types:
        for feature_name in FEATURE_CONFIG[feature_type]:
            print(f"Processing feature: {feature_name}")
            
            # Compare each model
            for i, model_dict in enumerate(model_dicts):
                df = convert_feature_df(gt_dict, model_dict, feature_name, feature_type)
                
                # Use AI model for scoring
                print(f"Scoring {model_names[i]}'s {feature_name} feature with {model_name}...")
                for j in tqdm(range(len(df))):
                    gt_feature = df['gt_feature'][j]
                    gen_feature = df['gen_feature'][j]
                    
                    # Format prompt
                    formatted_usr_prompt = usr_prompt_template.format(gt_feature, gen_feature)
                    
                    # Get AI score
                    response = get_one_response(formatted_usr_prompt, sys_prompt, model_name)
                    df.at[j, 'ai_evaluation'] = response
                
                # Save results
                output_file = os.path.join(output_dir, f"{model_names[i]}_{feature_name}_evaluation.csv")
                df.to_csv(output_file, index=False)
                print(f"Evaluation results saved to: {output_file}")
    
    print("Model output comparison complete.")




def compute_acc_mirco(gt: pd.Series, gen: pd.Series) -> float:
    '''
    Accuracy across all features
    '''
    gt_list = gt.tolist()
    gen_list = gen.tolist()
    correct = sum(1 for gt, gen in zip(gt_list, gen_list) if gt == gen)
    return round(correct / len(gt_list), 3) if gt_list else float("nan")

def compute_acc_macro(df: pd.DataFrame) -> float:
    '''
    Accuracy across all conditions
    '''
    conditions = df['condition'].unique()
    scores = []

    for condition in conditions:
        temp_df = df[df['condition'] == condition]
        gt_list = temp_df['gt_feature'].tolist()
        gen_list = temp_df['gen_feature'].tolist()
        score = (sum(1 for gt, gen in zip(gt_list, gen_list) if gt == gen)) / len(gt_list) if gt_list else float("nan")

        if not math.isnan(score):
            scores.append(score)

    return round(sum(scores) / len(scores), 3) if scores else float("nan")

def compute_f1_micro(df: pd.DataFrame, name: str) -> float:
    '''
    F1 across all reports
    '''
    grouped_df = df.groupby('study_id')
    scores = []

    for id, temp_df in grouped_df:
        gt_set = set(list(zip(temp_df['condition'], temp_df['gt_feature'])))
        gen_set = set(list(zip(temp_df['condition'], temp_df['gen_feature'])))

        assert len(gt_set) == len(gen_set), f"{id} {name} exists repetitive coniditons in gt/gen input"        

        tp = len(gt_set & gen_set)
        fp = len(gen_set - gt_set)
        fn = len(gt_set - gen_set)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)

    return round(sum(scores) / len(scores), 3) if scores else float("nan")

def compute_f1_macro(df: pd.DataFrame, name: str) -> float:
    '''
    Macro-averaged F1 across all conditions
    '''
    conditions = df['condition'].unique()
    scores = []

    for condition in conditions:
        condition_df = df[df['condition'] == condition]
        grouped_df = condition_df.groupby('study_id')
        for id, temp_df in grouped_df:
            gt_set = set(list(zip(temp_df['condition'], temp_df['gt_feature'])))
            gen_set = set(list(zip(temp_df['condition'], temp_df['gen_feature'])))

            assert len(gt_set) == len(gen_set), f"{condition} {name} exists repetitive conditions in gt/gen input"

            tp = len(gt_set & gen_set)
            fp = len(gen_set - gt_set)
            fn = len(gt_set - gen_set)

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            scores.append(f1)

    return round(sum(scores) / len(scores), 3) if scores else float("nan")

def preprocess(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def bleu_score(gt, gen):
    '''
    BLEU-4 computation
    '''
    smoothie = SmoothingFunction().method4
    return sentence_bleu([gt], gen, smoothing_function=smoothie)

def rouge_l_score(gt, gen):
    '''
    ROUGE-L computation
    '''
    gt_text = " ".join(gt)
    gen_text = " ".join(gen)
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return rouge.score(gt_text, gen_text)['rougeL'].fmeasure

def compute_similarity(gt_series, gen_series, metric='rouge'):
    '''
    Optimal matching: for each GT, find best Gen phrase
    '''
    all_scores = []
    for gt_list, gen_list in zip(gt_series, gen_series):
        scores = []
        for gt in gt_list:
            gt_tokens = preprocess(gt)
            if not gt_tokens:
                scores.append(0.0)
                continue

            best_score = 0.0
            for gen in gen_list:
                gen_tokens = preprocess(gen)
                if not gen_tokens:
                    continue

                if metric == 'bleu':
                    score = bleu_score(gt_tokens, gen_tokens)
                else:  # default: rouge
                    score = rouge_l_score(gt_tokens, gen_tokens)

                best_score = max(best_score, score)
            scores.append(best_score)
        all_scores.append(np.mean(scores))

    return round(np.mean(all_scores), 3)

def cal_metrics(df_feature, name, mode):
    '''
    Calculate appropriate metrics based on feature mode (QA or IE)
    '''
    if mode == 'QA':
        # 1. Acc. (micro)
        acc_micro = compute_acc_mirco(df_feature['gt_feature'], df_feature['gen_feature'])

        # 2. Acc. (macro)
        acc_macro = compute_acc_macro(df_feature)

        # 3. F1 (micro)
        f1_micro = compute_f1_micro(df_feature, name)

        # 4. F1 (macro)
        f1_macro = compute_f1_macro(df_feature, name)

        metric_dict = {
            'Feature': name,
            'Acc. (micro)': acc_micro,
            'Acc. (macro)': acc_macro,
            'F1 (micro)': f1_micro,
            'F1 (macro)': f1_macro            
        }
    elif mode == 'IE':
        # 1. ROUGE-L
        rouge = compute_similarity(df_feature['gt_feature'], df_feature['gen_feature'], metric='rouge')

        # 2. BLEU-4
        bleu = compute_similarity(df_feature['gt_feature'], df_feature['gen_feature'], metric='bleu')

        metric_dict = {
            'Feature': name,
            'ROUGE-L': rouge,
            'BLEU-4': bleu
        }
        
    return metric_dict

def evaluate_qa_features(dict_gt, dict_gen, metric_pth):
    '''
    Evaluate QA-type features
    '''
    print("Evaluating QA features...")
    metric_data = []
    mode = 'QA'
    
    for name in FEATURE_CONFIG[mode]:
        print(f"Processing {name}...")
        # 1. transform df
        df_feature = convert_feature_df(dict_gt, dict_gen, name, mode)

        # 2. calculate metrics
        metric_dict = cal_metrics(df_feature, name, mode)

        # 3. output metric
        metric_data.append(metric_dict)

    metric_df = pd.DataFrame(metric_data)
    os.makedirs(metric_pth, exist_ok=True)
    output_file = os.path.join(metric_pth, 'results_qa_avg.csv')
    metric_df.to_csv(output_file, index=False)
    print(f"QA evaluation results saved to {output_file}")
    return metric_df

def evaluate_ie_features(dict_gt, dict_gen, metric_pth):
    '''
    Evaluate IE-type features
    '''
    print("Evaluating IE features...")
    metric_data = []
    mode = 'IE'
    
    for name in FEATURE_CONFIG[mode]:
        print(f"Processing {name}...")
        # 1. transform df
        df_feature = convert_feature_df(dict_gt, dict_gen, name, mode)

        # 2. calculate metrics
        metric_dict = cal_metrics(df_feature, name, mode)

        # 3. output metric
        metric_data.append(metric_dict)

    metric_df = pd.DataFrame(metric_data)
    os.makedirs(metric_pth, exist_ok=True)
    output_file = os.path.join(metric_pth, 'results_ie_avg.csv')
    metric_df.to_csv(output_file, index=False)
    print(f"IE evaluation results saved to {output_file}")
    return metric_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate feature extraction metrics')
    parser.add_argument('--gen_path', type=str, required=True, 
                        help='Path to generated feature output JSON file')
    parser.add_argument('--gt_path', type=str, required=True, 
                        help='Path to ground truth feature JSON file')
    parser.add_argument('--metric_path', type=str, required=True, 
                        help='Directory to save metric results', default='./metrics')

    # New command line arguments for model comparison
    parser.add_argument('--compare_models', action='store_true',
                        help='Enable model comparison functionality')
    parser.add_argument('--model_paths', nargs='+', type=str,
                        help='Multiple model output paths to compare')
    parser.add_argument('--model_names', nargs='+', type=str,
                        help='Model names for output file naming')
    parser.add_argument('--feature_types', nargs='+', type=str, default=['IE'],
                        help='List of feature types to evaluate')
    parser.add_argument('--usr_prompt', type=str,
                        help='Path to user prompt template')
    parser.add_argument('--sys_prompt', type=str,
                        help='Path to system prompt template')
    parser.add_argument('--scoring_model', type=str, default='o1-mini',
                        help='Model name to use for scoring')                        
    
    args = parser.parse_args()
    
    if args.compare_models:
        # If model comparison is enabled
        if not args.model_paths:
            parser.error("--compare_models requires --model_paths")
        
        prompt_paths = {
            "user": args.usr_prompt if args.usr_prompt else "./usr_prompt.txt",
            "system": args.sys_prompt if args.sys_prompt else "./sys_prompt.txt"
        }
        
        compare_model_outputs(
            gt_path=args.gt_path,
            model_paths=args.model_paths,
            output_dir=args.metric_path,
            feature_types=args.feature_types,
            model_names=args.model_names,
            prompt_paths=prompt_paths,
            model_name=args.scoring_model
        )
    else:
        # Original evaluation functionality
        # Load input files
        print(f"Loading ground truth from {args.gt_path}...")
        with open(args.gt_path) as f:
            dict_gt = json.load(f)
            
        print(f"Loading generated features from {args.gen_path}...")
        with open(args.gen_path) as f:
            dict_gen = json.load(f)
        
        assert len(dict_gt) == len(dict_gen), "Length of ground truth and generated features mismatch!"
        
        # Evaluate features
        qa_results = evaluate_qa_features(dict_gt, dict_gen, args.metric_path)
        ie_results = evaluate_ie_features(dict_gt, dict_gen, args.metric_path)
        
        # Print summary results
        print("\nQA Evaluation Summary:")
        print(qa_results)
        
        print("\nIE Evaluation Summary:")
        print(ie_results)

if __name__ == "__main__":
    main()