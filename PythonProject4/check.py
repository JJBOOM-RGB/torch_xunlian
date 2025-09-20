from PIL import Image
import os

root = r'data_set/flower_data'          # 数据集根
bad = []
for split in ['train', 'val']:
    for cls in os.listdir(os.path.join(root, split)):
        folder = os.path.join(root, split, cls)
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                with Image.open(path) as im:
                    im.verify()      # 严格检查
            except Exception:
                bad.append(path)

print('共发现坏文件:', len(bad))
for p in bad:
    print(p)