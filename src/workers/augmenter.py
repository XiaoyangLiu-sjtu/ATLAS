import re
import gc
import torch
from vllm import LLM, SamplingParams
from utils import APIModel
import utils
    

class InfoviewExtracter:
    def _is_colon_in_brackets(self, string):
        stack = []
        bracket_pairs = {")": "(", "]": "[", "}": "{"}

        for char in string:
            if char in bracket_pairs.values():
                stack.append(char)
            elif char in bracket_pairs:
                if stack and stack[-1] == bracket_pairs[char]:
                    stack.pop()
            elif char == ":" and not stack:
                return False
        return True

    def _find_all_valid_colon_positions(self, string):
        positions = []
        stack = []
        bracket_pairs = {")": "(", "]": "[", "}": "{"}

        for i, char in enumerate(string):
            if char in bracket_pairs.values():
                stack.append(char)
            elif char in bracket_pairs:
                if stack and stack[-1] == bracket_pairs[char]:
                    stack.pop()
            elif char == ":" and not stack:
                if i + 1 < len(string) and string[i:i+2] != ":=":
                    positions.append(i)
        return positions

    def _split_line_by_colons(self, string):  
        valid_colons = self._find_all_valid_colon_positions(string)
        if len(valid_colons) <= 1:
            return [string] 
        
        parts = []
        start = 0
        for split_point in valid_colons[1:]:
                preceding_space = string.rfind(" ", start, split_point)
                preceding_word_space = string.rfind(" ", start, preceding_space)
                if preceding_word_space != -1:
                    parts.append(string[start:preceding_word_space].strip())
                    start = preceding_word_space  
        parts.append(string[start:].strip())
        return parts

    def extract_info(self, goal_string):
        # Regular expression matches "⊢" and its following content, and deletes "\\n" in it as theory_conclusion
        match = re.search(r"⊢(.*)", goal_string)
        theorem_conclusion = match.group(1).strip() if match else ""
        theorem_conclusion = theorem_conclusion.replace("\n", "")
        
        # Delete "\\n⊢" and the following content to get a new goal_string
        goal_string = re.sub(r"\n⊢.*", "", goal_string).strip()
        goal_string = re.sub(r"⊢.*", "", goal_string).strip()  
        
        # Use \n to split a string into multiple lines
        lines = goal_string.split("\n")
        
        # Initializes a list of storage conditions
        conditions = []
        current_condition = ""
        
        # Iterate over each line, detect if there are multiple colons, and process them
        for line in lines:
            parts = self._split_line_by_colons(line)  # Split multiple colons in a line
            for part in parts:
                if ":" in part and ":=" not in part and not self._is_colon_in_brackets(part):
                    if current_condition:  # If the current condition is not empty, add the previous condition first
                        conditions.append(current_condition.strip())
                    current_condition = part  # Update the current condition to this part
                else:
                    current_condition += " " + part.strip()  # Merge the current part into the previous condition
        
        # Add the last condition
        if current_condition:
            conditions.append(current_condition.strip())
        
        # Process each condition and remove extra spaces
        conditions = [condition.strip() for condition in conditions if condition.strip()]
        
        # # Create an empty list theorem_variables, traverse the elements of variables one by one, and find out which elements in variables contain "inst✝"
        # Get the part after ":" in this element, enclose it in square brackets and store it in theorem_variables. If it does not contain "inst✝", enclose it in round brackets and store it in theorem_variables
        theorem_variables = []
        for variable in conditions:
            if re.search(r"✝", variable):
                var_part = variable.split(":", 1)
                if len(var_part) > 1:
                    theorem_variables.append(f"[{var_part[1].strip()}]")  
                else:
                    theorem_variables.append(f"[{variable}]")  
            else:
                theorem_variables.append(f"({variable})")
        
        # Defining the structure of the theorem
        theorem_name = "tm_name"  
        theorem_variables = " ".join(variable for variable in theorem_variables)  
        theorem_conclusion = theorem_conclusion.strip()
        
        # Splice into a complete Lean theorem format and return information
        lean_theorem = f"theorem {theorem_name} {theorem_variables} : {theorem_conclusion} := by sorry"
        lean_theorem = lean_theorem.replace("\\\\", "\\")  # Change the escape characters to symbols with actual meaning
        lean_theorem = re.sub(r"✝", "", lean_theorem)  # Regular expression matching symbol "✝" and delete
        return lean_theorem
    

class FLParser:
    def _check_brackets_balanced(self, string):
        stack, bracket_pairs = [], {"(": ")", "[": "]", "{": "}"}
        for char in string:
            if char in bracket_pairs:
                stack.append(char)
            elif char in bracket_pairs.values():
                if not stack or bracket_pairs[stack.pop()] != char:
                    return False
        return not stack 

    def _find_colons_after_matching_parentheses(self, string):
        stack, brackets, colon_positions = [], {"(": ")", "[": "]", "{": "}"}, []
        for i, char in enumerate(string):
            if char in brackets:  
                stack.append(char)
            elif char in brackets.values(): 
                if stack and brackets[stack[-1]] == char:
                    stack.pop()
            elif char == ":" and not stack:  
                colon_positions.append(i)
        return colon_positions

    def _extract_brackets_content(self, string):
        stack, bracket_pairs = [], {"(": ")", "[": "]", "{": "}"}
        start, matches = -1, []
        for i, char in enumerate(string):
            if char in bracket_pairs:
                if not stack:
                    start = i
                stack.append(char)
            elif char in bracket_pairs.values():
                if stack and bracket_pairs[stack[-1]] == char:
                    stack.pop()
                    if not stack:
                        matches.append(string[start:i+1])
        return matches

    def parse_formal_statement(self, formal_statement):
        keyword_position = formal_statement.find("theorem")
        theorem_header = formal_statement[:keyword_position]
        theorem_content = formal_statement[keyword_position:]
        theorem_content = " ".join(theorem_content.split())
        split_theorem_content = theorem_content.split(" ")
        theorem_name = " ".join(split_theorem_content[:2])
        theorem_lines = split_theorem_content[2:]  

        theorem_variables_hypotheses, theorem_conclusion, temp = [], "", ""
        for index, line in enumerate(theorem_lines):
            temp = line if not temp else f"{temp} {line}" 
            balanced = self._check_brackets_balanced(temp) 
            if balanced:
                colon_mark = self._find_colons_after_matching_parentheses(temp) 
                if colon_mark == []:
                    matches = self._extract_brackets_content(temp)
                    theorem_variables_hypotheses.extend(matches)
                    temp = ""
                else:
                    matches = self._extract_brackets_content(temp[:colon_mark[0]])
                    theorem_variables_hypotheses.extend(matches)
                    theorem_conclusion += temp[colon_mark[0]:]
                    remaining_content = " ".join(theorem_lines[index + 1:])
                    if remaining_content:
                        theorem_conclusion += f" {remaining_content}"
                    break
            else:
                continue

        return {
            "theorem_header": theorem_header,
            "theorem_name": theorem_name,
            "theorem_variables_hypotheses": theorem_variables_hypotheses,
            "theorem_conclusion": theorem_conclusion+" by sorry",
        }

    def reorganize_formal_statement(self, data):
        for item in data:
            formal_statement = item["formal_statement"].replace(" := by sorry", " :=")
            parse_result = self.parse_formal_statement(formal_statement)
            reorganized_formal_statement = (
                parse_result["theorem_header"] + parse_result["theorem_name"] + "\n"
                + ("\n".join(parse_result["theorem_variables_hypotheses"]) + "\n" if parse_result["theorem_variables_hypotheses"] else "")
                + parse_result["theorem_conclusion"]
                ) 
            item["formal_statement"] = reorganized_formal_statement
        return data

    def parse_formal_statement_further(self, parse_result_list):
        temp_data_list = []
        for parse_result in parse_result_list:
            reorganized_formal_statement = (
                parse_result["theorem_header"] + parse_result["theorem_name"] + "\n"
                    + ("\n".join(parse_result["theorem_variables_hypotheses"]) + "\n" if parse_result["theorem_variables_hypotheses"] else "")
                    + parse_result["theorem_conclusion"]
                ).replace("sorry", "")

            count, tactic, theorem_varibles, theorem_hypotheses = 0, "", [], []
            for index, item in enumerate(parse_result["theorem_variables_hypotheses"]):
                if item.startswith("{") or item.startswith("["):
                    parse_result["theorem_variables_hypotheses"][index] = item+" | Type"
                else:
                    item = item.split(":", 1)[1][:-1]
                    tactic += "#check " + item + "\n"
                    count += 1
            temp_data_list.append({"formal_statement": reorganized_formal_statement, "tactic": tactic, "count": count, "result": parse_result})
        
        final_data_list = []
        outputs_list = utils.run_proofsteps(temp_data_list)
        for index, item in enumerate(outputs_list):
            mark_content = item["infos"]
            for i in range(temp_data_list[index]["count"]):
                try:
                    temp = mark_content[i]["data"]
                    last_colon_index = temp.rindex(" : ")
                    mark = temp[last_colon_index+3:].strip()
                except:
                    mark = "Type"
                for index_inner, item in enumerate(temp_data_list[index]["result"]["theorem_variables_hypotheses"]):
                    if item.endswith("Type") or item.endswith("Prop"):
                        continue
                    else:
                        if "Prop" in mark:
                            temp_data_list[index]["result"]["theorem_variables_hypotheses"][index_inner] = item+" | Prop"
                        else:
                            temp_data_list[index]["result"]["theorem_variables_hypotheses"][index_inner] = item+" | Type"
                        break
                
            theorem_varibles = [item.replace(" | Type", "") for item in temp_data_list[index]["result"]["theorem_variables_hypotheses"] if item.endswith("Type")]
            theorem_hypotheses = [item.replace(" | Prop", "") for item in temp_data_list[index]["result"]["theorem_variables_hypotheses"] if item.endswith("Prop")]
            final_data_list.append({
                "theorem_header": temp_data_list[index]["result"]["theorem_header"],
                "theorem_name": temp_data_list[index]["result"]["theorem_name"],
                "theorem_variables": theorem_varibles,
                "theorem_hypotheses": theorem_hypotheses,
                "theorem_conclusion": temp_data_list[index]["result"]["theorem_conclusion"]
            })
        return final_data_list

    def extract_hypotheses_name(self, result):
        hypotheses = result["theorem_hypotheses"]
        result["hypotheses_name"] = [
            item.split(":")[0].replace(" ", "").replace("(", "")
            for item in hypotheses
        ]
        return result

    def rewrite_type_brackets(self, file_path):
        data = utils.read_json(file_path)
        for index, item in enumerate(data):
            formal_statement = item["formal_statement"] if "formal_statement" in item else item["output"]
            formal_statement = formal_statement.replace(" by sorry", "").replace("\n", " ")
            parse_result = self.parse_formal_statement(formal_statement)
            for i, var_hyp in enumerate(parse_result["theorem_variables_hypotheses"]):
                patterns = [
                    (r'Type u(?:_\w+)?|Type v(?:_\w+)?|Type w(?:_\w+)?|Type x(?:_\w+)?|Type y(?:_\w+)?|Type z(?:_\w+)?', 'Type*'),
                    (r'Type _\)|Type\)', 'Type*)'),
                    (r'Type _\}|Type\}', 'Type*}'),
                    (r'Type _\]|Type\]', 'Type*]')
                ]
                temp = var_hyp
                for pattern, replacement in patterns:
                    temp = re.sub(pattern, replacement, temp)
                if "Type*" in temp:
                    temp = temp.replace("(", "{").replace(")", "}").replace("[", "{").replace("]", "}")
                    parse_result["theorem_variables_hypotheses"][i] = temp
            combined_value = (
                parse_result["theorem_header"] + parse_result["theorem_name"] + " " +
                (" ".join(parse_result["theorem_variables_hypotheses"]) + " " if parse_result["theorem_variables_hypotheses"] else "") +
                parse_result["theorem_conclusion"]
            )
            key = "formal_statement" if "formal_statement" in item else "output"
            item[key] = combined_value
        utils.write_json(file_path, data)

    
class FLAugmentation(FLParser, InfoviewExtracter):
    def __init__(self):
        super().__init__()

    def _generate_proofsteps_deepseek(self, formal_statement_list):  
        model = LLM(model="../Models/deepseek-ai/DeepSeek-Prover-V1.5-RL", tensor_parallel_size=1, trust_remote_code=True, dtype="bfloat16")
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=2048,
            top_p=0.95,
            n=1,
        )
        prompt = r"""Complete the following Lean4 code:\n\n```lean4\n"""
        model_inputs = [prompt+"import Mathlib\n"+formal_statement for formal_statement in formal_statement_list]
        model_outputs = model.generate(model_inputs, sampling_params=sampling_params)
        wholeproof_list = [[o.text for o in response.outputs] for response in model_outputs]

        proofsteplist_list = []
        for wholeproof in wholeproof_list:
            wholeproof = wholeproof[0]
            wholeproof = re.sub(r"/\-[\s\S]*?\-/", "", wholeproof)
            wholeproof = re.sub(r"--.*$", "", wholeproof, flags=re.MULTILINE)
            wholeproof = re.sub(r"\n\s*\n", "\n", wholeproof)
            wholeproof = wholeproof.split("\n")
            proofsteplist = [item.strip() for item in wholeproof if item.strip() != "```" and item.strip() != ""]
            proofsteplist_list.append(proofsteplist)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        return proofsteplist_list

    def augmentation_proof(self, formal_statement_list):
        proofsteplist_list = self._generate_proofsteps_deepseek(formal_statement_list)
        stepbystep_proofsteplist_list = [
            ["\n".join(proofsteplist[:i + 1]) for i in range(len(proofsteplist))]
            for proofsteplist in proofsteplist_list
        ]

        proof_proofstep_list = [
            {f"{formal_statement_list[index]}sorry": "import Mathlib\n" + formal_statement_list[index] + "\n" + stepbystep_proof}
            for index, stepbystep_proofsteplist in enumerate(stepbystep_proofsteplist_list)
            for stepbystep_proof in stepbystep_proofsteplist
        ]
        return proof_proofstep_list

    def augmentation_contraposition(self, formal_statement_list):
        parse_result_list = [
            self.parse_formal_statement(formal_statement.replace("by", ""))
            for formal_statement in formal_statement_list
        ]
        further_parse_result_list = self.parse_formal_statement_further(parse_result_list)
        
        contraposition_proofstep_list = [
            {f"{formal_statement_list[index]}sorry": "import Mathlib\n" + formal_statement_list[index] + "\ncontrapose! " + hypotheses_name + "\n"}
            for index, result in enumerate(further_parse_result_list)
            for hypotheses_name in self.extract_hypotheses_name(result)["hypotheses_name"]
        ]
        return contraposition_proofstep_list

    def extract_formal_statement(self, instruction, proofstep_list, result_augmentation, mark): 
        keys = [list(d.keys())[0] for d in proofstep_list]
        values = [list(d.values())[0] for d in proofstep_list]
        outputs_list = utils.verify_translation(formal_statements_list=values)

        for index, result in enumerate(outputs_list):
            errors = result.get("errors", [])
            for item in errors:
                data = item.get("data", "")
                if "unsolved goals" in data:
                    goals_string = data.replace("unsolved goals\n", "")
                    lean_theorem = self.extract_info(goals_string)
                    new_entry = {
                        "instruction": instruction,
                        "input": "null",
                        "output": lean_theorem,
                        "original_output": keys[index],
                    }
                    result_augmentation.append(new_entry)
        print(f"Extract_formal_statement Process for {mark} | Amount: {len(result_augmentation)}/{len(proofstep_list)}")
        return result_augmentation

    def compile_extract_formal_statement(self, result_augmentation, file_path, mark):  
        temp_data_list = ["import Mathlib\n" + result["output"] for result in result_augmentation]
        outputs_list = utils.verify_translation(formal_statements_list=temp_data_list)
        result_augmentation_pass = [
            result_augmentation[index] 
            for index, item in enumerate(outputs_list) 
            if item["pass"]
        ]
        utils.write_json(file_path, result_augmentation_pass)
        print(f"Compile_extract_formal_statement Process for {mark} | Amount: {len(result_augmentation_pass)}/{len(result_augmentation)}")


class FLTranslation(APIModel):
    def __init__(self, name_or_path, base_url, api_key, sampling_params, prompt):
        super().__init__(name_or_path, base_url, api_key, sampling_params)
        self.prompt = prompt

    def get_query(self, task):
        query = self.prompt.format(
            task
        )

        return [
                {"role": "system", "content": ""},
                {"role": "user", "content": query}
            ]