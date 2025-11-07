import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

word_hit_count=set()

def parse_json_file(json_file_expr):
    json_files_paths = sorted(glob.glob(json_file_expr))
    acc = None
    for json_files_path in json_files_paths:
        json_item = json.load(open(json_files_path))
        for t_name in json_item['results'].keys():
            if 'exact_match,flexible-extract' in  json_item['results'][t_name]:
                acc = json_item['results'][t_name]['exact_match,flexible-extract']
                break
            elif "pass@1,create_test" in json_item['results'][t_name]:
                acc = json_item['results'][t_name]["pass@1,create_test"]
                break

    assert acc is not None
    return acc


def get_score(input_dir, w_list, limit):
    weighted_avg_acc = []
    for w in w_list:
        json_files_expr = os.path.join(input_dir, f"gt_{limit}_{w}", ".*", "*.json")
        print(json_files_expr)
        json_files_paths = sorted(glob.glob(json_files_expr))
        if len(json_files_paths) >= 1:
            word_hit_count.add(w)
            #print(f"word hit:{w}, total:{len(word_hit_count)}")

            acc = parse_json_file(json_files_paths[0])
            weighted_avg_acc.append(acc)
    if len(weighted_avg_acc)  == len(w_list):
        return np.average(weighted_avg_acc)
    else:
        print("sample count is not enough")
        return None

def png_result(plot_data, plot_meta):
    plt.clf()
    plt.figure(figsize=(6,4))
    for plot_data_i in plot_data:
        plt.plot(plot_data_i["x"],
                 plot_data_i["y"],
                 #label=plot_data_i["label"],
                 linestyle=plot_data_i["linestyle"]
                 )
    plt.legend(fontsize=14)
    plt.ylim(0,plot_meta["ymax"])
    plt.xlabel("$\\ell$")
    plt.ylabel("Accuracy")
    plt.title(plot_meta["title_png"], fontsize=16)
    plt.savefig(plot_meta["fig_name_png"], bbox_inches="tight")

def tikz_result(plot_data, plot_meta):
    f = open(plot_meta["fig_name_tikz"], "w")
    output_str = ""
    output_str+= ("""
    \\begin{tikzpicture}
    \\begin{axis}[
        align=center,
        width=3.8cm,
        height=2.8cm,
        xlabel={$\\ell$},
        ylabel={Acc},
        legend pos=north west,
        grid=major,
        grid style={dashed,gray!30},
        xmin=0, xmax=%s,
        ymin=-0.05, ymax=%s,
        title={%s},
        title style={font=\\tiny},
        label style={font=\\tiny},
        tick label style={font=\\tiny},
        legend style={font=\\tiny},
        legend pos=outer north east,
        legend cell align={left},
        xlabel style={
            at={(current axis.south east)}, 
            anchor=north east,
            yshift=10pt,
            xshift=5pt
        },
        ylabel style={
            at={(current axis.north west)}, 
            anchor=north east,
            yshift=-20pt,
            xshift=20pt
        },
    ]
    """ % (plot_meta["xmax"], plot_meta["ymax"], plot_meta["title_tikz"].replace("_","\_")))
    for i, plot_data_i in enumerate(plot_data):
        thickness = "thick"
        if plot_data_i["linestyle"] == "dotted":
            thickness="very thick"
        output_str+= ("\\addplot[%s, %s, %s] table[row sep=\\\\] {\n" % (plot_data_i["color_tikz"], thickness, plot_data_i["linestyle"]))
        output_str+= ("  x y \\\\ \n")
        for x,y in zip(plot_data_i["x"],plot_data_i["y"]):
            output_str+= (f" {x} {y} \\\\ \n")
        output_str+= ("}; \n")
        #f.write("\\addlegendentry{%s}" % plot_data_i["label"])
    output_str+= ("""\\end{axis} \n \\end{tikzpicture} \n
    """)

    with open(plot_meta["fig_name_tikz"], "w") as f:
        f.write(output_str)

    with open(plot_meta["fig_name_tikz_2"], "w") as f:
        f.write(output_str)


def parse_all_result(input_dir, steer_dir, output_dir, s_limit=2, limit=20, dataset_name="gsm8k_adv", ymax=.7, model_name="qwen3b"):
    os.makedirs(output_dir,exist_ok=True)


    model_map={
        "MyQwen2.5-3B":"Qwen2.5-3B",
        "gemma-3-4b-it":"Gemma3-4B"
    }

    for top_k in [3,5, 8]:
        plot_data = []
        plot_meta = {}

        i = 0

        max_layer = 0
        for steer_key in [20, 21]:
            steer_type = f"steer_{s_limit}_{steer_key}"
            steer_files = sorted(glob.glob(os.path.join(steer_dir, steer_type, "word_*.txt")))
            all_scores = {}
            for fname in steer_files:
                try:
                    layer = int(os.path.basename(fname).split("_")[-1].replace(".txt",""))
                    max_layer = max(layer,max_layer)
                    if layer >=0:
                        w_list = ("".join(open(fname,"r").readlines())).strip().split("\n")
                        w_list = w_list[:top_k]
                        print(f"get score:{top_k}, {fname}")
                        score = get_score(input_dir, w_list, limit)
                        if score is not None:
                            all_scores[layer] = score
                except Exception as e:
                    pass
            all_scores = sorted(all_scores.items(),key=lambda x: x[0])
            plot_data.append({
               "x":[x[0] for x in all_scores],
               "y":[x[1] for x in all_scores],
               "linestyle": "solid",
                "color_tikz": f"c{11+i}"
            })
            i += 1

        gt_groups = {
            "L2": ["Wait", "Alternatively", "Check"],
            "L1": ["<|endoftext|>", "#", "%","<eos>"],
            "L0": ["Answer", "Result", "Output"],
        }
        for gkey, gwords in gt_groups.items():
            acc = []
            for gt_type in gwords:
                json_files_expr = os.path.join(input_dir.replace("step3","step1"), f"gt_{limit}_{gt_type}", ".*", "*.json")
                json_files_paths = sorted(glob.glob(json_files_expr))
                if len(json_files_paths) > 0:
                    acc.append(parse_json_file(json_files_paths[0]))
            plot_data.append({
                "x": [0,max_layer],
                "y": [np.average(acc)]*2,
                "label": f"baseline: {gkey}",
                "linestyle": "dotted",
                "color_tikz": f"c{11+i}"
            })
            i += 1

        steer_type = f"steer_baseline"
        fname = os.path.join(steer_dir, steer_type, "word_-1.txt")
        if os.path.exists(fname):
            w_list = ("".join(open(fname, "r").readlines())).strip().split("\n")
            w_list = w_list[:top_k]
            score = get_score(input_dir, w_list, limit)
            plot_data.append({
                "x": [0, max_layer],
                "y": [score] * 2,
                "label": "embedding",
                "linestyle": "dashed",
                "color_tikz": f"c{11 + i}"
            })
            i += 1

        plot_meta["title_png"]=f"Accuracy of Instructions in Dataset {dataset_name},\\\\Selected by Top-{top_k} CosSim"
        plot_meta["title_tikz"]=f"{model_map[model_name]}, Top-{top_k} \\\\ {dataset_name} "
        plot_meta["fig_name_png"]=os.path.join(output_dir,f"png_exp3_{model_name}_{dataset_name}_{top_k}.png")
        plot_meta["fig_name_tikz"]=os.path.join(output_dir,f"exp3_{model_name}_{dataset_name}_{top_k}.tex")
        plot_meta["fig_name_tikz_2"]=os.path.join("all_exp3_latex_fig",f"exp3_{model_name}_{dataset_name}_{top_k}.tex")
        plot_meta["ymax"] = ymax
        plot_meta["xmax"] = max_layer

        png_result(plot_data, plot_meta)
        tikz_result(plot_data, plot_meta)


def run():
    model_name="MyQwen2.5-3B"
    parse_all_result(input_dir=f"visualize/gsm8k_adv/{model_name}/step3" ,
                     steer_dir=f"visualize/gsm8k_adv/{model_name}/step2" ,
                    output_dir=f"visualize/gsm8k_adv/{model_name}/step5",
                     s_limit=200,
                     limit=2000,
                     ymax=.6,
                     dataset_name="gsm8k_adv",
                     model_name=model_name
                     )

    model_name="gemma-3-4b-it"
    parse_all_result(input_dir=f"visualize/gsm8k_adv/{model_name}/step3" ,
                     steer_dir=f"visualize/gsm8k_adv/{model_name}/step2" ,
                     output_dir=f"visualize/gsm8k_adv/{model_name}/step5",
                     s_limit=200,
                     limit=2000,
                     ymax=.9,
                     dataset_name="gsm8k_adv",
                     model_name=model_name
                     )


    model_name="MyQwen2.5-3B"
    dataset_name="cruxeval_o_adv"
    parse_all_result(input_dir=f"visualize/{dataset_name}/{model_name}/step3" ,
                     steer_dir=f"visualize/{dataset_name}/{model_name}/step2" ,
                     output_dir=f"visualize/{dataset_name}/{model_name}/step5",
                     s_limit=200,
                     limit=500,
                     ymax=.2,
                     dataset_name=dataset_name,
                     model_name=model_name
                     )

    #model_name="MyQwen2.5-3B"
    model_name="gemma-3-4b-it"
    dataset_name="cruxeval_o_adv"
    parse_all_result(input_dir=f"visualize/{dataset_name}/{model_name}/step3" ,
                     steer_dir=f"visualize/{dataset_name}/{model_name}/step2" ,
                     output_dir=f"visualize/{dataset_name}/{model_name}/step5",
                     s_limit=200,
                     limit=500,
                     ymax=.4,
                     dataset_name=dataset_name,
                     model_name=model_name
                     )



if __name__ == '__main__':
    run()
