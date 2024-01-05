from super_gradients.training import models
from super_gradients.common.object_names import Models
import cv2
import os
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage
from PIL import Image
import torch
import time
import shutil
from loguru import logger


def ensure_multiple_of_eight(value):
    remainder = value % 16
    if remainder != 0:
        value += (16 - remainder)
    return value


def resize_img_short_size(img, short_size = 1280):
    w,h = img.shape[1], img.shape[0]
    img_short_size = min(w, h)
    scale = short_size / img_short_size
    scale_w = int(scale * w)
    scale_h = int(scale * h)
    out_size = (ensure_multiple_of_eight(scale_w), ensure_multiple_of_eight(scale_h))
    print(f'out_size {out_size}')
    img = cv2.resize(img, out_size)
    return img  

def img_pre_process(img_path):
    pre_proccess = Compose([
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    img = cv2.imread(img_path)
    img = resize_img_short_size(img)
    img = pre_proccess(img).unsqueeze(0).cuda()
    return img

checkpoint_path = '/opt/ml/new_ml/super-gradients/notebook_ckpts/regseg_jcheng/RUN_20240104_185036_464401/ckpt_best.pth'

model = models.get(model_name=Models.REGSEG48_T, num_classes=1, checkpoint_path=checkpoint_path)
model = model.to('cuda')
model = model.eval()

# perf
# start = time.time()
# for i in range(1000):
#     mask = model(img)
# end = time.time()
# print(f"infer time: {end - start}")

shutil.rmtree('infer_results', ignore_errors=True)
os.makedirs('infer_results', exist_ok=True)

validExts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

need_run_val = True
if need_run_val:
    valid_img_dir = '/opt/ml/datasets/jiaocheng/valid'
   
    img_list = os.listdir(valid_img_dir)
    for img_name in img_list:
        ext = img_name[img_name.rfind("."):].lower()
        if not ext.endswith(validExts):
            continue
        img_path = os.path.join(valid_img_dir, img_name)
        logger.info("img path {}", img_path)
        original_image = cv2.imread(img_path)
        original_image = resize_img_short_size(original_image)
        img = img_pre_process(img_path)
        mask = model(img)
        mask = torch.sigmoid(mask).gt(0.5).squeeze()
        mask = ToPILImage()(mask.float())
        mask_np = np.array(mask)
       
        mask_np_rgb = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        mask_np_rgb[mask_np == 255] = [0, 0, 255]
        mask_np_rgb = cv2.resize(mask_np_rgb, (original_image.shape[1], original_image.shape[0]))
        merged_image = cv2.addWeighted(original_image, 0.7, mask_np_rgb, 0.3, 0)
        # mask.save(f"infer_results/infer_{img_name}.png")
        cv2.imwrite(f"infer_results/{img_name}", merged_image)
