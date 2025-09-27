import os
import re
import gc
import time
import math
import torch
import json
import yaml
import subprocess
import numpy as np
from Levenshtein import distance
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from src.workers.verifier import FLVerifier


class APIModel:
    def __init__(self, name_or_path, base_url, api_key, sampling_params):
        self.name_or_path = name_or_path
        self.base_url = base_url
        self.api_key = api_key
        self.sampling_params = sampling_params
        self.client = self._init_model()

    def _init_model(self):
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=10000,
            max_retries=10,
        )

    def generate(self, task):
        try:
            response = self.client.chat.completions.create(
                model=self.name_or_path,
                messages=self.get_query(task),
                **self.sampling_params,
            )
            return response.choices[0].message.content
        except:
            return "null"

    def generate_batch(self, task_list, desc, max_workers=100):
        remain_task_list, responses, resolve_times = task_list, [], 0
        while remain_task_list:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.generate, task): task for task in remain_task_list}
                for future in tqdm(futures, total=len(remain_task_list), desc=desc): 
                    task = futures[future]
                    result = future.result()
                    if extract_text(result) != "null":
                        responses.append((task, extract_text(result)))
            remain_task_list = [item for item in remain_task_list if item not in [task for task, _ in responses]]
            print(f"Remaining tasks: {len(remain_task_list)}")

            resolve_times += 1
            if resolve_times >= 20:
                print("Too many retries, stopping...")
                break
        return responses
    
    def __del__(self):
        del self.client
        gc.collect()
    

class LocalModel:
    def __init__(self, name_or_path, gpus, seed, sampling_params, prompt):
        self.name_or_path = name_or_path
        self.gpus = gpus
        self.seed = seed
        self.sampling_params = sampling_params
        self.prompt = prompt
        self._init_model()

    def _init_model(self):
        self.model = LLM(
            model=self.name_or_path,
            tensor_parallel_size=self.gpus,
            trust_remote_code=True,
            dtype="bfloat16",
            max_num_seqs=64,
            seed=self.seed
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path, trust_remote_code=True)

    def generate_batch(self, task_list, system_content=""):
        queries = [self.get_query(task, system_content) for task in task_list]
        responses = self.model.generate(queries, sampling_params=SamplingParams(**self.sampling_params))
        return [[o.text for o in response.outputs] for response in responses]

    def __del__(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def merge_json(file_path, json_paths):
    data = [item for json_path in json_paths for item in read_json(json_path)]
    write_json(file_path, data)


def extract_text(result):
    # start_index = result.find("# Answer")
    # if start_index != -1:
    #     result = result[start_index:] 
    # else:
    #     return "null"
    match = re.search(r"\|\|(.*?)\|\|", result, re.DOTALL)
    if match:
        return match.group(1)
    return "null"


def remove_informal_prefix(formal_statement):
    pattern = r'/-- .*? -/\n'
    cleaned_text = re.sub(pattern, '', formal_statement, flags=re.DOTALL)
    return cleaned_text


def unified_prefix(file_path):
    data = read_json(file_path)
    for item in data:
        if item["compiler_check"] == "success":
            formal_statement, item["ori_formal_statement"] = item["formal_statement"], item["formal_statement"]
            theorem_position, lemma_position = formal_statement.find("theorem"), formal_statement.find("lemma")
            example_position, def_position = formal_statement.find("example"), formal_statement.find("def")
            if theorem_position != -1:
                formal_statement = formal_statement[theorem_position:]
                parts = formal_statement.split(maxsplit=2)
                item["formal_statement"] = "theorem tm_name" + (' ' + parts[2] if len(parts) > 2 else '')
            elif lemma_position != -1:
                formal_statement = formal_statement[lemma_position:]
                parts = formal_statement.split(maxsplit=2)
                item["formal_statement"] = "theorem tm_name" + (' ' + parts[2] if len(parts) > 2 else '')
            elif example_position != -1:
                formal_statement = formal_statement[example_position:]
                parts = formal_statement.split(maxsplit=2)
                item["formal_statement"] = "theorem tm_name" + (' ' + parts[2] if len(parts) > 2 else '')
            elif def_position != -1:
                formal_statement = formal_statement[def_position:]
                parts = formal_statement.split(maxsplit=2)
                item["formal_statement"] = "theorem tm_name" + (' ' + parts[2] if len(parts) > 2 else '')
            else:
                item["formal_statement"] = formal_statement
    write_json(file_path, data)
    return data


def update_data_with_responses(data, responses, key_to_locate, key_to_update, file_path):
    key_to_index = {item[key_to_locate]: i for i, item in enumerate(data)}
    for response in responses:
        key_value = response[0][key_to_locate]
        if key_value in key_to_index:
            if key_to_update == "input":
                data[key_to_index[key_value]][key_to_update] = response[1].replace("Theorem: ", "")
            elif key_to_update == "formal_statement":
                stmt = response[1]
                keyword_position = stmt.find("theorem")
                if keyword_position != -1:
                    stmt = stmt[keyword_position:]
                    parts = stmt.split(maxsplit=2)
                    data[key_to_index[key_value]][key_to_update] = "theorem tm_name" + (' ' + parts[2] if len(parts) > 2 else '')
                else:
                    data[key_to_index[key_value]][key_to_update] = "theorem tm_name := by sorry"
            else:
                data[key_to_index[key_value]][key_to_update] = response[1]
    write_json(file_path, data)


def load_verification(file_path):
    data = read_json(file_path)
    formal_statements_list = [
        "import Mathlib\n" + item["formal_statement"] for item in data
    ]
    return formal_statements_list


def update_verification(file_path, key, results):
    data = read_json(file_path)
    for item, result in zip(data, results):
        item[key] = "fail" if result is False else ("success" if result is True else result)
    write_json(file_path, data)


def verify_translation(file_path="", formal_statements_list="", key="", verified_file_path=""):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if not formal_statements_list:
        formal_statements_list = load_verification(file_path)

    start_time = time.time()
    lean4_scheduler = FLVerifier()
    request_id_list = lean4_scheduler.submit_all_request(formal_statements_list)
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    lean4_scheduler.close()
    end_time = time.time()
    print(f"Verification time: {end_time - start_time:.2f} seconds")

    if key:
        update_verification(file_path, key, [item["pass"] for item in outputs_list])
    if verified_file_path:
        write_json(verified_file_path, outputs_list)
    return outputs_list


def run_proofsteps(data):
    formal_statements_list = [
        f"import Mathlib\n{item['formal_statement']}\n{item['tactic']}"
        for item in data
    ]
    outputs_list = verify_translation(formal_statements_list=formal_statements_list)
    return outputs_list


def extract_split_data(file_path, checked_file_path, remained_file_path):
    data = read_json(file_path)
    result_checked, result_remained = [], []
    for item in data:
        if "nli_check" in item and item["nli_check"] in ["good", "average"]:
            new_entry = {
                "instruction": "Translate the natural language statement to Lean4 code:\n**Informal statement**",
                "input": item["informal_statement"],
                "output": item["formal_statement"].replace("\n", " ")
            }
            result_checked.append(new_entry)
        else:
            new_entry = {
                "id": item["id"],
                "concept1": item["concept1"],
                "domain1": item["domain1"],
                "concept2": item["concept2"],
                "domain2": item["domain2"],
                "informal_statement": item["informal_statement"]
            }
            result_remained.append(new_entry)
    write_json(checked_file_path, result_checked)
    write_json(remained_file_path, result_remained)


def find_most_different(original_output, candidates):
    distances = [distance(original_output, candidate) for candidate in candidates]
    max_index = np.argmax(distances)
    return candidates[max_index]


def filter_only(file_path):
    data = read_json(file_path)
    grouped = {}
    for entry in data:
        original_output = entry["original_output"]
        if original_output not in grouped:
            grouped[original_output] = []
        grouped[original_output].append(entry["output"])
    
    result = []
    for original_output, candidates in grouped.items():
        if "contraposition" in file_path._str:
            most_different_output = find_most_different(original_output, candidates)
        elif "proof" in file_path._str:
            most_different_output = candidates[-1]

        for entry in data:
            if entry["output"] == most_different_output and entry["original_output"] == original_output:
                entry_copy = entry.copy()
                entry_copy.pop("original_output", None) 
                result.append(entry_copy)
                break
    write_json(file_path, result)


def summarize_results_generation(file_path, proof_augmentation_file_path, contraposition_augmentation_file_path):
    data, proof_augmentation_data, contraposition_augmentation_data = read_json(file_path), read_json(proof_augmentation_file_path), read_json(contraposition_augmentation_file_path)
    compiler_check_translation_count, compiler_check_revision_count = 0, 0
    nli_check_translation_count, nli_check_revision_count = 0, 0

    for item in data:
        is_nli_check_passed = item.get("nli_check") in ["good", "average"]
        if item["compiler_check_translation"] == "success" and item["compiler_check_revision"] == "success":
            compiler_check_translation_count += 1
            if is_nli_check_passed:
                nli_check_translation_count += 1
        if item["compiler_check_translation"] == "fail" and item["compiler_check_revision"] == "success":
            compiler_check_revision_count += 1
            if is_nli_check_passed:
                nli_check_revision_count += 1
    
    statistics = {
        "Initial_Data_Amount": len(data),
        "Remained_Data_Amount": len(data) - nli_check_translation_count - nli_check_revision_count,
        "Total_Data_Amount": nli_check_translation_count+nli_check_revision_count+len(proof_augmentation_data+contraposition_augmentation_data),
        "First_Translation_Amount_Ratio": [compiler_check_translation_count, round(compiler_check_translation_count/len(data)*100, 2)],
        "Second_Translation_Amount_Ratio": [compiler_check_revision_count, round(compiler_check_revision_count/(len(data)-compiler_check_translation_count)*100, 2)],
        "Synthetic_Data_Amount_First_Second": [nli_check_translation_count+nli_check_revision_count, nli_check_translation_count, nli_check_revision_count],
        "Augmented_Data_Amount_Proof_Contraposition": [len(proof_augmentation_data+contraposition_augmentation_data), len(proof_augmentation_data), len(contraposition_augmentation_data)]
    }
    return list(statistics.values())


def write_markdown(file_path, content_list):
    with open(file_path, "w", encoding="utf-8") as file:
        for content in content_list:
            file.write(content)
            file.write("\n")


def read_yaml(file_path):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def write_yaml(file_path, data):
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def modify_configs(round_index):
    data = {
    f"dataset{round_index}": {
        "file_name": f"/nfs/my/lxy/ATLAS/paper/dataset/iteration_data/dataset_{round_index}/dataset_total.json",
        "columns": {
          "prompt": "instruction",
          "query": "input",
          "response": "output"
        }
      }
    }
    write_json("../LLaMA-Factory/data/dataset_info.json", data)

    temp_data = read_json(f"/nfs/my/lxy/ATLAS/paper/dataset/iteration_data/dataset_{round_index}/dataset_total.json")
    data = read_yaml("../LLaMA-Factory/configs/finetune.yaml")
    data["model_name_or_path"] = f"../Models/ATLAS_Translator_{round_index-1}"
    data["dataset"] = f"dataset{round_index}"
    data["output_dir"] = f"saves/ATLAS_Translator_{round_index}"
    data["gradient_accumulation_steps"] = math.floor(128*(len(temp_data))/56830/4)
    write_yaml("../LLaMA-Factory/configs/finetune.yaml", data)

    data = read_yaml("../LLaMA-Factory/configs/merge.yaml")
    data["model_name_or_path"] = f"../Models/ATLAS_Translator_{round_index-1}"
    data["adapter_name_or_path"] = f"saves/ATLAS_Translator_{round_index}"
    data["export_dir"] = f"../Models/ATLAS_Translator_{round_index}"
    write_yaml("../LLaMA-Factory/configs/merge.yaml", data)


def is_tmux_running(tmux_name):
    try:
        result = subprocess.run(["tmux", "ls"], capture_output=True, text=True, check=True)
        sessions = [line.split(':')[0] for line in result.stdout.splitlines()]
        return tmux_name in sessions
    except subprocess.CalledProcessError:
        return False
    

def finetune_merge(round_index):
    modify_configs(round_index)
    subprocess.run(["tmux", "new-session", "-d", "-s", "finetune"])
    path_command = "cd ../LLaMA-Factory"
    activate_command = "conda activate LLaMA-Factory"
    finetune_command = "llamafactory-cli train configs/finetune.yaml"
    merge_command = "llamafactory-cli export configs/merge.yaml"
    full_command = f"{path_command} && {activate_command} && {finetune_command} && {merge_command}"
    tmux_exec_command = f"tmux send-keys -t finetune '{full_command} && exit' Enter"
    subprocess.run(tmux_exec_command, shell=True)
    time.sleep(100)
    while is_tmux_running("finetune"):
        time.sleep(10)
    print("Finetune and merge completed.")


def merge_data(data_a, data_b, file_path):
    result, block_size = [], 16
    i, j = 0, 0
    len_a, len_b = len(data_a), len(data_b)
    while i < len_a or j < len_b:
        result.extend(data_a[i:i + block_size])
        result.extend(data_b[j:j + block_size])
        i += block_size
        j += block_size
    write_json(file_path, result)


def summarize_results_evaluation(file_path, number_to_generate):
    compiler_check_count, nli_check_count, beq_check_count = 0, 0, 0
    total_groups, current_group = 0, []
    data = read_json(file_path)

    for item in data:
        current_group.append(item)
        if len(current_group) == number_to_generate:
            if any(item.get("compiler_check") == "success" for item in current_group):
                compiler_check_count += 1
            if any(item.get("nli_check") == "same" for item in current_group):
                nli_check_count += 1
            if any(item.get("beq_check") == "success" for item in current_group):
                beq_check_count += 1
            total_groups += 1
            current_group = []
    return [round(compiler_check_count / total_groups * 100, 2), round(nli_check_count / total_groups * 100, 2), round(beq_check_count / total_groups * 100, 2)]