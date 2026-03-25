import os
import matplotlib.pyplot as plt
import numpy as np

image_path = "/data/hongrui/Killifish/CLI/results/1/predicted_mask_test"
base_path = "/data/hongrui/Killifish/CLI/results/1/predicted_mask_error"


plt_dict = {}
count_dict = {}

for folder_name in os.listdir(base_path):
    plt_dict[folder_name] = {}

    for small_folder_name in os.listdir(image_path):
        plt_dict[folder_name][small_folder_name] = {}

        if small_folder_name in os.listdir(os.path.join(base_path, folder_name)):
            for file_name in os.listdir(os.path.join(base_path, folder_name, small_folder_name)):
                if file_name.split("_")[0] not in plt_dict[folder_name][small_folder_name]:
                    plt_dict[folder_name][small_folder_name][file_name.split("_")[0]] = {}

                if file_name.split(".png")[0].endswith("mask"):
                    plt_dict[folder_name][small_folder_name][file_name.split("_mask.png")[0]]["mask"] = os.path.join(
                        base_path, folder_name, small_folder_name, file_name)
                else:
                    plt_dict[folder_name][small_folder_name][file_name.split("_input.png")[0]]["input"] = os.path.join(
                        base_path, folder_name, small_folder_name, file_name)
            plt_dict[folder_name][small_folder_name]["len"] = len(plt_dict[folder_name][small_folder_name])
        else:
            plt_dict[folder_name][small_folder_name]["len"] = 0

for folder_name in os.listdir(image_path):
    count_dict[folder_name] = len(os.listdir(os.path.join(image_path, folder_name))) / 2

all_days = set()
for day in plt_dict:
    all_days.update(int(key.split()[1]) for key in os.listdir(image_path))

sorted_days = sorted(all_days)

order = ["below_5_percent", "between_5_and_10_percent", "above_10_percent"]
for key in order:
    x = []
    y = []
    for day in sorted_days:
        matched_key = next((k for k in plt_dict[key] if int(k.split()[1]) == day), None)
        x.append(day)
        y.append(plt_dict[key][matched_key]['len'])

    if key == "above_10_percent":
        key = "    > 10%"
        color = "red"
    elif key == "below_5_percent":
        key = "    <  5%"
        color = "green"
    else:
        key = "5% ≤ x ≤ 10%"
        color = "blue"
    plt.plot(x, y, marker='.', label=key, color=color)

plt.xlabel("Day")
plt.ylabel("Error num")
plt.title("Length vs. Time")
plt.legend()
plt.grid(True)
plt.show()

bottoms = np.zeros(len(sorted_days))
for key in order:
    y = []
    for day in sorted_days:
        matched_key = next((k for k in plt_dict[key] if int(k.split()[1]) == day), None)
        y.append(plt_dict[key][matched_key]['len'] / count_dict[matched_key] * 100)

    if key == "above_10_percent":
        key = "    > 10%"
        color = "red"
    elif key == "below_5_percent":
        key = "    <  5%"
        color = "green"
    else:
        key = "5% ≤ x ≤ 10%"
        color = "blue"

    plt.bar(sorted_days, y, bottom=bottoms, width=7, label=key, color=color)
    bottoms += y

plt.xlabel("Day")
plt.ylabel('Percentage (%)')
plt.title('Accuracy per Day')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

bottoms = np.zeros(len(sorted_days))
for key in order:
    y = []
    for day in sorted_days:
        matched_key = next((k for k in plt_dict[key] if int(k.split()[1]) == day), None)
        y.append(plt_dict[key][matched_key]['len'])

    if key == "above_10_percent":
        key = "    > 10%"
        color = "red"
    elif key == "below_5_percent":
        key = "    <  5%"
        color = "green"
    else:
        key = "5% ≤ x ≤ 10%"
        color = "blue"

    plt.bar(sorted_days, y, bottom=bottoms, width=7, label=key, color=color)
    bottoms += y

plt.xlabel("Day")
plt.ylabel('Count')
plt.title('Accuracy per Day')

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
