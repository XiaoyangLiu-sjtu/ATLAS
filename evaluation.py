import json
from src.workers.synthesizer import NLTranslation
from src.workers.back_translator import BackTranslation
from src.workers.nli_checker import NLICheck
import configs.config_evaluation as ceval
import utils


def benchmark_translation():
    model_sampling_params = ceval.model_sampling_params.get(model, ceval.model_sampling_params["ATLAS_Translator_Series"])
    model_sampling_params["n"] = number_to_generate
    nltranslation = NLTranslation(
        ceval.model_name_or_path[model], 
        ceval.model_gpus, 
        seed,
        model_sampling_params,
        ceval.Prompt_NL_Translation.get(model, ceval.Prompt_NL_Translation["ATLAS_Translator_Series"])
    )  

    with open(f"benchmarks/{benchmark}.jsonl", "r", encoding="utf-8") as file:
        task_list = [{"informal_statement": json.loads(line)["informal_statement"]} for line in file]
    if model == "HERALD_Translator":
        extra_prompt = "You are an expert at Lean 4 and Mathematics."
    elif model == "Kimina-Autoformalizer-7B":
        extra_prompt = "You are an expert in mathematics and Lean 4."
    else:
        extra_prompt = None
    responses = nltranslation.generate_batch(task_list, extra_prompt) if extra_prompt else nltranslation.generate_batch(task_list)
    del nltranslation

    results = [
        {
            "id": f"{model}_{benchmark}_{i+1}_{j+1}",
            "informal_statement": task_list[i]["informal_statement"],
            "formal_statement": utils.remove_informal_prefix(formal_statement) if model == "HERALD_Translator" else formal_statement
        }
        for i, response in enumerate(responses)
        for j, formal_statement in enumerate(response)
    ]
    utils.write_json(file_path, results)


def back_translation():
    backtranslation = BackTranslation(
        ceval.Back_Translation_model_name_or_path, 
        ceval.Back_Translation_model_gpus,
        seed,
        ceval.Back_Translation_model_sampling_params, 
        ceval.Prompt_Back_Translation
    )

    if model == "HERALD_Translator" or model == "Kimina-Autoformalizer-7B" or model == "DeepSeek-V3":
        data = utils.unified_prefix(file_path)
    else:
        data = utils.read_json(file_path)
    success_indices, task_list = [], [] 
    for idx, item in enumerate(data):
        if item.get("compiler_check") == "success":
            task_list.append({"formal_statement": item["formal_statement"]})
            success_indices.append(idx)
    responses = backtranslation.generate_batch(task_list)
    del backtranslation

    for i, response in enumerate(responses):
        data[success_indices[i]]["back_translation"] = response[0]
    utils.write_json(file_path, data)


def nli_check():
    nlicheck = NLICheck(
        ceval.NLI_Check_model_name_or_path,
        ceval.NLI_Check_model_base_url, 
        ceval.NLI_Check_model_api_key, 
        ceval.NLI_Check_model_sampling_params, 
        ceval.Prompt_Semantic_Contrast
    )
    
    data = utils.read_json(file_path)
    task_list = [
        {
            "id": item["id"],
            "informal_statement": item["informal_statement"],
            "back_translation": item["back_translation"]
        }
        for item in data
        if item.get("back_translation") and item.get("compiler_check") == "success"
    ]
    responses = nlicheck.generate_batch(task_list, desc="NLI Check")
    del nlicheck

    utils.update_data_with_responses(data, responses, "id", "nli_check", file_path)


def main():
    benchmark_translation()
    utils.verify_translation(file_path=file_path, key="compiler_check")
    back_translation()
    nli_check()


if __name__ == "__main__":
    models = ["HERALD_Translator", "Kimina-Autoformalizer", "ATLAS_Translator*_Q"]
    seeds = [42, 43, 44, 45, 46]
    benchmarks = ["minif2f", "proofnet", "putnam", "mathqual", "connf"]
    number_to_generates = [1, 8, 32]

    for model in models:
        for seed in seeds:
            markdown_file_path = f"paper/experiment/main_experiment/{model}/seed{seed}/experiment_results.md"
            header = "|Model|Benchmark|Pass@k|Compiler_Check|NLI_Check|BEq_Check|"
            split_line = "|-|-|-|-|-|-|"
            markdown_content_list = [header, split_line]
            for benchmark in benchmarks:
                for number_to_generate in number_to_generates:
                    file_path = f"paper/experiment/main_experiment/{model}/seed{seed}/{benchmark}_{number_to_generate}.json"
                    main()
                    statistics = utils.summarize_results_evaluation(file_path, number_to_generate)
                    markdown_content_list.append(f"|{model}|{benchmark}|{number_to_generate}|{'|'.join(map(str, statistics))}|")
                    utils.write_markdown(markdown_file_path, markdown_content_list)
                    print(f"Completed: {model}, {benchmark}, {number_to_generate}")