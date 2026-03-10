import argparse
import json
import os.path

def get_doc_cruxeval_o(item):
    doc_truth_cot = item["cot"]
    doc_messy_cot = item["messy_cot"]
    doc = {
        "code": item["code"],
        "input": item["input"],
        "output": item["output"],
        "question_id": item["question_id"],
        "cot_truth": doc_truth_cot,
        "cot_messy": doc_messy_cot,
    }
    return doc

def get_doc_gsm8k(item):
    doc_truth_cot = item["original_cot"]
    doc_messy_cot = item["messy_cot"]
    doc = {
        "input": item["question"],
        "output": item["answer"],
        "question_id": item["question_id"],
        "cot_truth": doc_truth_cot,
        "cot_messy": doc_messy_cot,
    }
    return doc


def run(args):
    json_output_dir = os.path.join(args.visualize_dir, args.json_out_name.replace(".json",""), "step0")
    json_output_path = os.path.join(json_output_dir, args.json_out_name)


    os.makedirs(json_output_dir, exist_ok=True)
    input_filename = args.input_file
    out_line = []
    items = json.load(open(input_filename, "r"))
    for item0 in items:
        if "cruxeval_o" in args.input_file:
            input_type = "cruxeval_o"
            doc = get_doc_cruxeval_o(item0)
        elif "gsm8k" in args.input_file:
            input_type = "gsm8k"
            doc = get_doc_gsm8k(item0)
        else:
            assert 0
        print(doc["question_id"])
        doc["original_question_id"] = doc["question_id"]
        item0["original_question_id"] = doc["original_question_id"]
        doc["question_id"] = len(out_line)
        item0["question_id"] = doc["question_id"]
        doc["input_type"] = input_type

        out_line.append(doc)


    json.dump(out_line, open(json_output_path, "w"), indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default="mydataset/cruxeval_o_adv/train.json")
    parser.add_argument("--json_out_name", type=str, default="cruxeval_o_adv.json")
    parser.add_argument("--visualize_dir", type=str, default="visualize")
    args = parser.parse_args()
    run(args)
