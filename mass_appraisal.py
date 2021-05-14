import os


def alter(file, old_str, new_str):
    with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:

            if old_str in line:
                line = line.replace(old_str, new_str)

            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)


yolo = r"./yolo.py"
get_map = r"./get_map.py"
for idx in range(1, 4):
    old_pth = "Epoch" + str(idx) + ".pth"
    new_pth = "Epoch" + str(idx + 1) + ".pth"
    old_result = 'results/results_' + str(idx)
    new_result = 'results/results_' + str(idx + 1)
    os.system("python ./get_dr_txt.py")
    os.system("python ./get_gt_txt.py")
    os.system("python ./get_map.py")
    alter(yolo, old_pth, new_pth)
    alter(get_map, old_result, new_result)
alter(yolo, "Epoch4.pth", "Epoch1.pth")
alter(get_map, 'results/results_4', 'results/results_1')
