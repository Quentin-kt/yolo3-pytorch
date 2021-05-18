import os
import torch
import numpy as np

"""
 损失函数——多标签二分类交叉熵（Binary Cross Entropy）
"""
# 初始化
classes = ["ChJ", "BX", "ZhAW", "CK", "QF", "TJ", "JG", "ShG", "PL", "ZhGAJ", "CQBG", "YWChR", "ShL", "FSh"]


# 将文件夹内所有txt文件转换为一个tensor
def file_lines_to_tensor(dir_path, confidence_whether):
    total_file_count = sum([len(files) for root, dirs, files in os.walk(dir_path)])
    numpy = np.zeros((total_file_count, len(classes)))
    for filepath, dir_path, filenames in os.walk(dir_path):
        idx = -1
        for name in filenames:
            # 初始化下标和最大概率
            idx += 1
            max_confidence = np.zeros(len(classes))
            with open(os.path.join(filepath, name), "r", encoding="utf-8") as txt:
                for line in txt:
                    for idx_class in range(len(classes)):
                        if classes[idx_class] in line:
                            if confidence_whether:
                                confidence = float(line.split()[1])
                                if confidence > max_confidence[idx_class]:
                                    max_confidence[idx_class] = confidence
                            else:
                                max_confidence[idx_class] = 1
                            break
            numpy[idx][:] = max_confidence
    tensor = torch.from_numpy(numpy)
    return tensor


detection = file_lines_to_tensor(r'input/detection-results', True)
truth = file_lines_to_tensor(r'input/ground-truth', False)
loss = torch.nn.BCELoss()
print(loss(detection, truth))