import os
import sys
# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import argparse
import re
from vllm import LLM, SamplingParams
import json
from configs.models import MODEL_CONFIGS
from configs.prompts import SYS_PROMPT_5, USR_PROMPT_3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, dest='model_name', 
                        default=None, 
                        help="Model name to select from MODEL_CONFIGS.")
    parser.add_argument("--input_csv", type=str, 
                        dest="input_csv", default=None, 
                        help="Path to input CSV file containing reports.")
    parser.add_argument("--o", type=str, 
                        dest='output_dir', default=None, 
                        help="Directory path to output results.")
    parser.add_argument("--prompt", type=str, 
                        dest="prompt", default="5",
                        help="Prompt version to use (e.g., 5 for zero-shot).")
    args = parser.parse_known_args()

    return args


class vLLMProcessor:
    def __init__(self, model_name, input_csv, output_dir, prompt="5"):
        self.model = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.prompt_label = prompt
        self.in_csv = input_csv
        self.out_dir = output_dir
        self.sampling_params, self.tokenizer, self.llm = self.prepare_llm()

    def prepare_llm(self):
        '''
        Initialize vLLM model
        '''
        sampling_params = SamplingParams(temperature=1e-5, max_tokens=4096)
        llm = LLM(self.config["model_path"], tensor_parallel_size=self.config["tensor_parallel_size"])
        tokenizer = llm.get_tokenizer()

        return sampling_params, tokenizer, llm

    def get_prompt(self, prompt_type=''):
        '''
        Return prompt from imported constants
        '''
        if prompt_type == 'sys':
            return SYS_PROMPT_5
        elif prompt_type == 'usr':
            return USR_PROMPT_3
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

    def run_label_extraction(self, ls_id, df_gt_repo):
        '''
        Extract labels using zero-shot prompting
        '''
        all_prompt = []
        prompt_s = self.get_prompt(prompt_type='sys')  # 使用导入的系统提示
        
        for i in range(len(ls_id)):
            id = ls_id[i]
            # use report_y as prompt_u
            prompt_u = list(df_gt_repo[df_gt_repo['study_id']==id]['report_y'])[0] if 'report_y' in df_gt_repo.columns else ''

            all_prompt.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": prompt_s},
                        {"role": "user", "content": prompt_u},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        
        ls_all_outputs = self.llm.generate(all_prompt, self.sampling_params)
        return ls_all_outputs

    def run(self):
        '''
        Main execution method
        '''
        # Step 1: Load Files
        print("Loading input data...")
        df_gt_repo = pd.read_csv(self.in_csv)
        df_gt_repo['study_id'] = df_gt_repo['study_id'].apply(lambda x: str(x))
        df_gt_repo = df_gt_repo.sort_values(by='study_id').reset_index(drop=True)
        ls_id = list(np.unique(df_gt_repo['study_id']))
        task1_results = {}

        # Step 2: Label Extraction
        print("Running label extraction...")
        ls_all_outputs = self.run_label_extraction(ls_id, df_gt_repo)

        assert len(ls_all_outputs) == len(ls_id)
        print('Finished label extraction.')
        print('Processing output...')

        # Step 3: Extract results
        for i in range(len(ls_id)):
            id = ls_id[i]
            generated_text = ls_all_outputs[i].outputs[0].text
            task1_match = re.search(r'<TASK1>(.*?)</TASK1>', generated_text, re.DOTALL)
            
            if task1_match:
                task1_content = task1_match.group(1).strip()
                # 假设模型输出格式总是正确的
                task1_results[id] = json.loads(task1_content)
            else:
                print(f"Warning: No TASK1 match for ID {id}")
                task1_results[id] = generated_text  # 保存原始文本作为备用
        
        # Step 4: Save files
        print("Saving results...")
        os.makedirs(self.out_dir, exist_ok=True)
        output_file = os.path.join(self.out_dir, f'gt_results_{self.prompt_label}_cleaned.json')
        with open(output_file, 'w', encoding='utf-8') as task1_file:
            json.dump(task1_results, task1_file, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")


if __name__ == '__main__':
    args, _ = parse_args()

    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{args.model_name}' not found in MODEL_CONFIGS.")

    if not args.input_csv:
        raise ValueError("Input CSV file must be specified with --input_csv")
    
    if not args.output_dir:
        raise ValueError("Output directory must be specified with --o")

    processor = vLLMProcessor(
        model_name=args.model_name,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        prompt=args.prompt
    )
    processor.run()