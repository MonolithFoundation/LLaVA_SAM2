import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def show_anns(ori_img, anns, borders=True):
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
    if len(anns) == 0:
        return

    if ori_img.shape[2] == 3:
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2RGBA)

    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

    mask_img = np.zeros_like(ori_img)

    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.randint(0, 256, 3), [128]])
        mask_img[m] = color_mask

        if borders:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # 尝试平滑轮廓
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(mask_img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    combined_img = cv2.addWeighted(ori_img, 0.5, mask_img, 0.5, 0.5)

    cv2.imshow("Annotated Image", combined_img)
    cv2.imshow("Origin Image", ori_img)
    cv2.imshow("Mask", mask_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = Image.open("notebooks/images/groceries.jpg")
image = np.array(image.convert("RGB"))
print(image.shape)


# sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"

sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cpu", apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)
masks = mask_generator.generate(image)

# plt.figure(figsize=(20, 20))
# plt.imshow(image)
show_anns(image, masks)
# plt.axis("off")
# plt.show()
