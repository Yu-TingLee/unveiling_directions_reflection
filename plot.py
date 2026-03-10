import argparse
import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def _load_result(path):
    item = json.load(open(path))
    for t in item["results"].values():
        if "exact_match,flexible-extract" in t:
            return t["exact_match,flexible-extract"]
        if "pass@1,create_test" in t:
            return t["pass@1,create_test"]
    assert False, f"no recognised metric in {path}"


def _png(plot_data, plot_meta):
    plt.clf()
    plt.figure(figsize=(6, 4))
    for d in plot_data:
        plt.plot(d["x"], d["y"], label=d["label"], linestyle=d["linestyle"])
    plt.legend(fontsize=14)
    plt.ylim(0, plot_meta["ymax"])
    plt.xlabel("$\\ell$")
    plt.ylabel("Accuracy")
    plt.title(plot_meta["title_png"], fontsize=16)
    plt.savefig(plot_meta["fig_name_png"], bbox_inches="tight")


def _tikz(plot_data, plot_meta):
    ymax = plot_meta["ymax"]
    ytick = "0,0.2,0.4,0.6,0.8,1" if ymax > 0.5 else "0,0.1,0.2,0.3,0.4,0.5"
    out = """    \\begin{tikzpicture}
    \\begin{axis}[
        align=center,
        width=3.8cm,
        height=2.8cm,
        xlabel={$\\ell$},
        ylabel={Acc},
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
        ytick={%s}
    ]
    """ % (plot_meta["xmax"], ymax, plot_meta["title_tikz"].replace("_", r"\_"), ytick)
    for d in plot_data:
        thickness = "very thick" if d["linestyle"] == "dotted" else "thick"
        out += "\\addplot[%s, %s, %s] table[row sep=\\\\] {\n" % (d["color_tikz"], thickness, d["linestyle"])
        out += "  x y \\\\ \n"
        for x, y in zip(d["x"], d["y"]):
            out += f" {x} {y} \\\\ \n"
        out += "}; \n"
    out += "\\end{axis} \n \\end{tikzpicture} \n"
    for fname in (plot_meta["fig_name_tikz"], plot_meta["fig_name_tikz_2"]):
        with open(fname, "w") as f:
            f.write(out)


# ---- instruction selection ------------------------------------

def _avg_score(input_dir, w_list, limit):
    accs = []
    for w in w_list:
        paths = sorted(glob.glob(os.path.join(input_dir, f"gt_{limit}_{w}", ".*", "*.json")))
        if paths:
            accs.append(_load_result(paths[0]))
    return np.average(accs) if len(accs) == len(w_list) else None


def _plot_exp3(input_dir, steer_dir, output_dir, s_limit, limit, dataset_name, ymax, model_name, latex_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)

    model_map = {"MyQwen2.5-3B": "Qwen2.5-3B", "MyGemma-3-4B-it": "Gemma3-4B"}

    for top_k in [3, 5, 8]:
        plot_data = []
        max_layer = 0
        i = 0

        for steer_key in [20, 21]:
            steer_type = f"steer_{s_limit}_{steer_key}"
            steer_files = sorted(glob.glob(os.path.join(steer_dir, steer_type, "word_*.txt")))
            scores = {}
            for fname in steer_files:
                try:
                    layer = int(os.path.basename(fname).split("_")[-1].replace(".txt", ""))
                    max_layer = max(layer, max_layer)
                    if layer >= 0:
                        w_list = open(fname).read().strip().split("\n")[:top_k]
                        score = _avg_score(input_dir, w_list, limit)
                        if score is not None:
                            scores[layer] = score
                except Exception:
                    pass
            scores = sorted(scores.items(), key=lambda x: x[0])
            plot_data.append({
                "x": [x[0] for x in scores],
                "y": [x[1] for x in scores],
                "label": f"steer_{steer_key}",
                "linestyle": "solid",
                "color_tikz": f"c{11 + i}",
            })
            i += 1

        gt_groups = {
            "L2": ["Wait", "Alternatively", "Check"],
            "L1": ["<|endoftext|>", "#", "%", "<eos>"],
            "L0": ["Answer", "Result", "Output"],
        }
        for gkey, gwords in gt_groups.items():
            acc = []
            for gt in gwords:
                paths = sorted(glob.glob(os.path.join(
                    input_dir.replace("step3", "step1"), f"gt_{limit}_{gt}", ".*", "*.json")))
                if paths:
                    acc.append(_load_result(paths[0]))
            plot_data.append({
                "x": [0, max_layer], "y": [np.average(acc)] * 2,
                "label": f"baseline: {gkey}", "linestyle": "dotted",
                "color_tikz": f"c{11 + i}",
            })
            i += 1

        baseline_file = os.path.join(steer_dir, "steer_baseline", "word_-1.txt")
        if os.path.exists(baseline_file):
            w_list = open(baseline_file).read().strip().split("\n")[:top_k]
            score = _avg_score(input_dir, w_list, limit)
            plot_data.append({
                "x": [0, max_layer], "y": [score] * 2,
                "label": "embedding", "linestyle": "dashed",
                "color_tikz": f"c{11 + i}",
            })
            i += 1

        plot_meta = {
            "title_png": f"Accuracy of Instructions in Dataset {dataset_name},\\\\ Selected by Top-{top_k} CosSim",
            "title_tikz": f"{model_map[model_name]}, Top-{top_k} \\\\ {dataset_name}",
            "fig_name_png": os.path.join(output_dir, f"exp3_{model_name}_{dataset_name}_{top_k}.png"),
            "fig_name_tikz": os.path.join(output_dir, f"exp3_{model_name}_{dataset_name}_{top_k}.tex"),
            "fig_name_tikz_2": os.path.join(latex_dir, f"exp3_{model_name}_{dataset_name}_{top_k}.tex"),
            "ymax": ymax,
            "xmax": max_layer,
        }
        _png(plot_data, plot_meta)
        _tikz(plot_data, plot_meta)


# ---- activation steering --------------------------------------

def _plot_exp4(dataset_name, input_dir, output_dir, ymax, limit, model_name, latex_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)

    if model_name == "MyQwen2.5-3B":
        skey_stype = {
            1:  ["Output", "Result", "#", "%"],
            -1: ["Alternatively", "Check", "#", "%"],
        }
    elif model_name == "MyGemma-3-4B-it":
        skey_stype = {
            1:  ["Output", "Result", "#", "%"],
            -1: ["Alternatively", "Check", "#", "%"],
        }
    else:
        assert False, f"unknown model: {model_name}"

    model_map = {"MyQwen2.5-3B": "Qwen2.5-3B", "MyGemma-3-4B-it": "Gemma3-4B"}
    max_layers = {"MyQwen2.5-3B": 35, "MyGemma-3-4B-it": 33}

    for s, start_types in skey_stype.items():
        if s == 1:
            steering_vector_name = {"20_tselected": "Level2 - Level0", "21_tselected": "Level2 - Level1"}
        else:
            steering_vector_name = {"20_tselected": "Level2 - Level0", "10_tselected": "Level1 - Level0"}

        for start_type in start_types:
            plot_data = []
            i = 0
            layer_ids = list(range(max_layers[model_name] + 1))

            for steering_type, label in steering_vector_name.items():
                results = []
                layer_id_exists = []
                for layer_id in layer_ids:
                    paths = sorted(glob.glob(os.path.join(
                        input_dir, f"s{s}_{start_type}_{limit}_{steering_type}_l{layer_id}", ".*", "*.json")))
                    if paths:
                        layer_id_exists.append(layer_id)
                        results.append(_load_result(paths[0]))
                if layer_id_exists:
                    plot_data.append({
                        "x": layer_id_exists, "y": results,
                        "label": label, "linestyle": "solid",
                        "color_tikz": f"c{11 + i}",
                    })
                    i += 1

            gt_groups = {
                "L2": ["Wait", "Alternatively", "Check"],
                "L1": ["<|endoftext|>", "#", "%", "<eos>"],
                "L0": ["Answer", "Result", "Output"],
            }
            for gkey, gwords in gt_groups.items():
                acc = []
                for gt in gwords:
                    paths = sorted(glob.glob(os.path.join(
                        input_dir.replace("step4", "step1"), f"gt_{limit}_{gt}", ".*", "*.json")))
                    if paths:
                        acc.append(_load_result(paths[0]))
                plot_data.append({
                    "x": [0, max_layers[model_name]], "y": [np.average(acc)] * 2,
                    "label": f"baseline: {gkey}", "linestyle": "dotted",
                    "color_tikz": f"c{11 + i}",
                })
                i += 1

            st = start_type.replace("<eos>", "eos").replace("%", "p")
            plot_meta = {
                "title_png": f"{model_map[model_name]},\\\\ {dataset_name},\\\\ Intervene `{start_type.replace('%', r'\%').replace('#', r'\#')}'",
                "title_tikz": f"{model_map[model_name]},\\\\ {dataset_name},\\\\ Intervene `{start_type.replace('%', r'\%').replace('#', r'\#')}'",
                "fig_name_png": os.path.join(output_dir, f"exp4_{model_name}_{dataset_name}_{s}_{st}.png"),
                "fig_name_tikz": os.path.join(output_dir, f"exp4_{model_name}_{dataset_name}_{s}_{st}.tex"),
                "fig_name_tikz_2": os.path.join(latex_dir, f"exp4_{model_name}_{dataset_name}_{s}_{st}.tex"),
                "ymax": ymax,
                "xmax": max_layers[model_name],
            }
            _png(plot_data, plot_meta)
            _tikz(plot_data, plot_meta)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize_dir", type=str, default="visualize")
    parser.add_argument("--latex_dir_exp3", type=str, default="all_exp3_latex_fig")
    parser.add_argument("--latex_dir_exp4", type=str, default="all_exp4_latex_fig")
    parser.add_argument("--model_names", nargs="+", default=["MyQwen2.5-3B", "MyGemma-3-4B-it"])
    parser.add_argument("--dataset_names", nargs="+", default=["gsm8k_adv", "cruxeval_o_adv"])
    parser.add_argument("--s_limit", type=int, default=200)
    parser.add_argument("--limits", nargs="+", type=int, default=[2000, 500])
    parser.add_argument("--ymax_config", type=str, default=None,
                        help='JSON dict mapping "model,dataset" to ymax, e.g. \'{"MyQwen2.5-3B,gsm8k_adv": 0.6}\'')
    args = parser.parse_args()

    default_ymax_exp3 = {
        ("MyQwen2.5-3B",  "gsm8k_adv"):      0.6,
        ("MyGemma-3-4B-it", "gsm8k_adv"):      0.9,
        ("MyQwen2.5-3B",  "cruxeval_o_adv"): 0.2,
        ("MyGemma-3-4B-it", "cruxeval_o_adv"): 0.4,
    }
    default_ymax_exp4 = {
        ("MyQwen2.5-3B",  "gsm8k_adv"):      0.6,
        ("MyGemma-3-4B-it", "gsm8k_adv"):      0.9,
        ("MyQwen2.5-3B",  "cruxeval_o_adv"): 0.13,
        ("MyGemma-3-4B-it", "cruxeval_o_adv"): 0.5,
    }

    if args.ymax_config:
        override = {tuple(k.split(",", 1)): v for k, v in json.loads(args.ymax_config).items()}
        ymax_exp3 = override
        ymax_exp4 = override
    else:
        ymax_exp3 = default_ymax_exp3
        ymax_exp4 = default_ymax_exp4

    dataset_limits = dict(zip(args.dataset_names, args.limits))

    for dataset_name in args.dataset_names:
        limit = dataset_limits[dataset_name]
        for model_name in args.model_names:
            _plot_exp3(
                input_dir=os.path.join(args.visualize_dir, dataset_name, model_name, "step3"),
                steer_dir=os.path.join(args.visualize_dir, dataset_name, model_name, "step2"),
                output_dir=os.path.join(args.visualize_dir, dataset_name, model_name, "step5"),
                s_limit=args.s_limit,
                limit=limit,
                ymax=ymax_exp3.get((model_name, dataset_name), 0.7),
                dataset_name=dataset_name,
                model_name=model_name,
                latex_dir=args.latex_dir_exp3,
            )
            _plot_exp4(
                dataset_name=dataset_name,
                input_dir=os.path.join(args.visualize_dir, dataset_name, model_name, "step4"),
                output_dir=os.path.join(args.visualize_dir, dataset_name, model_name, "step6"),
                ymax=ymax_exp4.get((model_name, dataset_name), 0.7),
                limit=limit,
                model_name=model_name,
                latex_dir=args.latex_dir_exp4,
            )


if __name__ == "__main__":
    run()
