import sys
import matplotlib.pyplot as plt
from tueplots.constants.color import rgb, palettes
import json

sys.path.append('../galvani')
from misc_util import get_next_tue_plot_color
from misc_util import get_config as load_results


def minmax_norm(vector):
    _min = min(vector)
    _max = max(vector)
    result = [(x-_min)/(_max-_min) for x in vector]
    return result


if __name__ == "__main__":
    results = []
    for eval_idx in range(1,9):
        config_path = f"../../out/eval{eval_idx}/cfg/eval_metrics_config.json"
        eval_dict = load_results(config_path)
        results.append(eval_dict)

    SAVE_DIR = "../../latex/report_template/figures/"
    FIG_W = 15
    FIG_H = 5
    AXES_ASPECT = 10
    FONTSIZE = 15
    BW = 0.1
    LW = 0.2
    GRID_LW = 0.5
    LEGEND_OFFSET_X = 1.13
    LEGEND_OFFSET_Y = 0.99

    for image in ["left", "right"]:
        for metric_distribution_characteristic_idx in [0, 1]:
            metric_distribution_characteristic = "mean" if metric_distribution_characteristic_idx == 0 else "std"

            metrics_labels = ["PSNR", "SSIM", "LPIPS"]
            scores_per_metric = {"PSNR": [], "SSIM": [], "LPIPS": []}
            for eval_idx in range(1,9):
                eval_dict = results[eval_idx-1]
                for metric in metrics_labels:
                    scores_per_metric[metric].append(eval_dict[f"mean_std_{metric.lower()}_{image}"][metric_distribution_characteristic_idx])

            for metric in metrics_labels:
                scores_per_metric[metric] = minmax_norm(scores_per_metric[metric])   

            fig, ax = plt.subplots(figsize=(FIG_W,FIG_H))
            bar_xpos = range(3)
            label_xpos = None
            for eval_idx in range(1,9):
                if eval_idx != 6:
                    ax.bar(
                        bar_xpos,
                        [scores_per_metric[key][eval_idx-1] for key in scores_per_metric],
                        width=BW,
                        color=get_next_tue_plot_color(eval_idx-1),
                        align="edge",
                        label=f"eval{eval_idx}"
                    )
                    if eval_idx == 5: label_xpos = [pos-(BW/2) for pos in bar_xpos]
                    bar_xpos = [pos+BW for pos in bar_xpos]

            title_prefix = "Average" if metric_distribution_characteristic == "mean" else "Std. of"
            ax.set_title(f"{title_prefix} generation quality scores ({image} image)", fontsize=FONTSIZE)
            ax.set_ylabel(f"normalised {metric_distribution_characteristic}", fontsize=FONTSIZE)
            
            ax.legend(
                bbox_to_anchor=(LEGEND_OFFSET_X, LEGEND_OFFSET_Y), 
                fontsize=FONTSIZE
            ).get_frame().set_edgecolor(color=rgb.tue_gray)

            plt.xticks(label_xpos, metrics_labels, fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)

            # fig.tight_layout()
            fig.savefig(SAVE_DIR+f"eval_quality_{image}_{metric_distribution_characteristic}.pdf")


    FIG_W = 13
    LEGEND_OFFSET_X = 1.138

    scores_per_eval = []
    for i in range(8):
        eval_dict = results[i]
        if i in [6, 7]:
            scores_per_eval.append([
                1.0, # dirty trick ;)
                eval_dict[f"mean_std_ssim_right"][0]
            ])
        else:
            scores_per_eval.append([
                eval_dict[f"mean_std_ssim_left"][0],
                eval_dict[f"mean_std_ssim_right"][0]
            ])

    fig, ax = plt.subplots(figsize=(FIG_W,FIG_H))
    bar_xpos = range(7) # -eval6
    label_xpos = None
    for image in [0, 1]:
        ax.bar(
            bar_xpos,
            [scores_per_eval[eval_idx-1][image] for eval_idx in [1, 2, 3, 4, 5, 7, 8]],
            width=BW,
            color=get_next_tue_plot_color(image),
            align="edge",
            label="left" if image == 0 else "right"
        )
        if image == 1: label_xpos = bar_xpos
        bar_xpos = [pos+BW for pos in bar_xpos]

    ax.set_title("Structural Similarity Index of left/right images compared to their resp. ground truth", fontsize=FONTSIZE)
    ax.set_ylabel(f"average SSIM", fontsize=FONTSIZE)
    
    ax.legend(
        bbox_to_anchor=(LEGEND_OFFSET_X, LEGEND_OFFSET_Y), 
        fontsize=FONTSIZE
    ).get_frame().set_edgecolor(color=rgb.tue_gray)

    plt.xticks(label_xpos, [f"eval{eval_idx}" for eval_idx in [1, 2, 3, 4, 5, 7, 8]], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    # fig.tight_layout()
    fig.savefig(SAVE_DIR+f"ssim_left_vs_right.pdf")
