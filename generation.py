import re
import random
from pathlib import Path
import src.workers.synthesizer as synthesizer
import src.workers.augmenter as augmenter
import configs.config_generation as cgen
import utils


def nl_generation():
    NLGeneration = synthesizer.NLGeneration(
        cgen.Teacher_model_name_or_path,
        cgen.Teacher_model_base_url,
        cgen.Teacher_model_api_key,
        cgen.Teacher_model_sampling_params,
        cgen.Prompt_NL_Generation)
    
    task_list = [random.sample(utils.read_json("configs/concept_repository.json"), 2) for _ in range(1)]
    responses = NLGeneration.generate_batch(task_list, "NL Generation")
    del NLGeneration

    results = [
        {
            "id": f"{round_index}_{index+1}",
            "concept1": response[0][0]["concept"],
            "domain1": response[0][0]["domain"],
            "concept2": response[0][1]["concept"],
            "domain2": response[0][1]["domain"],
            "informal_statement": response[1].replace("Theorem: ", "").strip(),
        }
        for index, response in enumerate(responses)
    ]
    utils.write_json(file_path, results)

    remained_file_path = Path(f"paper/dataset/iteration_data/dataset_{round_index-1}/dataset_remained.json")
    if remained_file_path.exists():
        utils.merge_json(file_path, [remained_file_path, file_path])


def nl_translation():
    NLTranslation = synthesizer.NLTranslation(
        cgen.Student_model_name_or_path, 
        cgen.Student_model_gpus, 
        cgen.Student_model_sampling_params, 
        cgen.Prompt_NL_Translation
    )

    data = utils.read_json(file_path)
    task_list = [
        {
            "informal_statement": item["informal_statement"],
        }
        for item in data
        ]
    responses = NLTranslation.generate_batch(task_list)
    del NLTranslation

    for i, response in enumerate(responses):
        data[i]["formal_statement"] = response[0]
    data = utils.normalize_data(data, responses)
    data = FLParser.reorganize_formal_statement(data)
    utils.write_json(file_path, data)
    utils.verify_translation(file_path=file_path, key="compiler_check_translation", verified_file_path=compiler_check_translation_path)


def nl_revision():
    FLRevision = synthesizer.FLRevision(
        cgen.Teacher_model_name_or_path,
        cgen.Teacher_model_base_url,
        cgen.Teacher_model_api_key,
        cgen.Teacher_model_sampling_params,
        cgen.Prompt_FL_Revision
    )

    data = utils.read_json(file_path)
    verified_data = utils.read_json(compiler_check_translation_path)
    task_list = [
        {
            "id": item["id"],
            "formal_statement": f"import Mathlib\n{item['formal_statement']}",
            "compiler_error_messages": verified_data[i].get("errors"),
            "informal_statement": item["informal_statement"],
        }
        for i, item in enumerate(data) 
        if item["compiler_check_translation"] == "fail"
    ]
    responses = FLRevision.generate_batch(task_list, "FL Revision")
    del FLRevision

    utils.update_data_with_responses(data, responses, "id", "formal_statement", file_path)
    FLParser.rewrite_type_brackets(file_path)
    utils.verify_translation(file_path=file_path, key="compiler_check_revision")


def nl_alignment():
    FLAlignment = synthesizer.FLAlignment(
        cgen.Teacher_model_name_or_path, 
        cgen.Teacher_model_base_url, 
        cgen.Teacher_model_api_key, 
        cgen.Teacher_model_sampling_params, 
        cgen.Prompt_FL_Alignment
    )

    data = utils.read_json(file_path)
    task_list = [
        {
            "id": item["id"],
            "formal_statement": item["formal_statement"],
            "informal_statement": item["informal_statement"],
        }
        for item in data
        if item["compiler_check_revision"] == "success"
    ]
    responses = FLAlignment.generate_batch(task_list, desc="FL Alignment")
    del FLAlignment

    utils.update_data_with_responses(data, responses, "id", "nli_check", file_path)
    utils.extract_split_data(file_path, checked_file_path, remained_file_path)


def fl_augmentation():
    FLAugmentation = augmenter.FLAugmentation()
    result_checked = utils.read_json(checked_file_path)
    instruction = result_checked[0]["instruction"]
    formal_statement_list = [re.sub("sorry", "", item["output"]) for item in result_checked]

    methods = [
        (FLAugmentation.augmentation_proof, "proof", [], proof_augmentation_file_path),
        (FLAugmentation.augmentation_contraposition, "contraposition", [], contraposition_augmentation_file_path)
    ]
    for aug_func, tag, result_list, save_path in methods:
        proofstep_list = aug_func(formal_statement_list)
        result_list = FLAugmentation.extract_formal_statement(instruction, proofstep_list, result_list, tag)
        FLAugmentation.compile_extract_formal_statement(result_list, save_path, tag)
        utils.filter_only(save_path)
        FLParser.rewrite_type_brackets(save_path)


def fl_translation():
    FLTranstion = synthesizer.FLTranslation(
        cgen.Teacher_model_name_or_path, 
        cgen.Teacher_model_base_url, 
        cgen.Teacher_model_api_key, 
        cgen.Teacher_model_sampling_params, 
        cgen.Prompt_FL_Translation
    )
    
    methods = [proof_augmentation_file_path, contraposition_augmentation_file_path]
    for file_path in methods:
        data = utils.read_json(file_path)
        task_list = [
            {
                "output": item["output"]
            }
            for item in data
        ]
        responses = FLTranstion.generate_batch(task_list, "FL Translation")
        utils.update_data_with_responses(data, responses, "output", "input", file_path)
    del FLTranstion

    utils.merge_json(total_file_path, [checked_file_path, proof_augmentation_file_path, contraposition_augmentation_file_path])


def main():
    nl_generation()
    nl_translation()
    nl_revision()
    nl_alignment()
    fl_augmentation()  
    fl_translation()

    statistics = utils.summarize_results_generation(file_path, proof_augmentation_file_path, contraposition_augmentation_file_path)
    header = "|Initial_Data_Amount|Remained_Data_Amount|Total_Data_Amount|First_Translation_Amount_Ratio|Second_Translation_Amount_Ratio|Synthetic_Data_Amount_First_Second|Augmented_Data_Amount_Proof_Contraposition|"
    split_line = "|-|-|-|-|-|-|-|"
    row = f"|{'|'.join(map(str, statistics))}|"
    utils.write_markdown(markdown_file_path, [header, split_line, row])
    print(f"Generation Completed: Round {round_index} | Next for finetune")
    utils.finetune_merge(round_index)


if __name__ == "__main__":
    for round_index in range(1, 11):
        base_path = Path(f"paper/dataset/iteration_data/dataset_{round_index}")
        file_path = base_path / "dataset.json"
        compiler_check_translation_path = base_path / "dataset_compiler_check_translation.json"
        checked_file_path = base_path / "dataset_checked.json"
        remained_file_path = base_path / "dataset_remained.json"
        proof_augmentation_file_path = base_path / "dataset_proof_augmentation.json"
        contraposition_augmentation_file_path = base_path / "dataset_contraposition_augmentation.json"
        total_file_path = base_path / "dataset_total.json"
        markdown_file_path = base_path / "experiment_results.md"

        FLParser = augmenter.FLParser()
        cgen.Student_model_name_or_path = f"../Models/ATLAS_Translator_{round_index-1}"
        main()