import torch
import pickle
import numpy as np
import cv2
import chardet
import re
import matplotlib.pyplot as plt
import numpy as np

with open("./plt/warp","rb") as f:
    data = pickle.load(f)

# for psv in range(7):
#     # 直接绘图
#     # cv2.imwrite("./plt/psv1_{}.png".format(psv),seg[psv]*255)

#     #密度图
#     fig = plt.figure(figsize=(12, 6))
#     plt.hist(data[psv,0].flatten(), bins=5)
#     # plt.show()

#     # 热力图
#     # data = (seg[psv]*255).astype(int)

#     # 循环数据并创建文本注释
#     # ha 水平对齐方式
#     # for i in range(540):
#     #     for j in range(960):
#     #         text = ax.text(j, i, '{:.1f}'.format(data[i, j]),
#     #                     ha="center", va="center", color="w")

#     # fig.tight_layout()
#     plt.savefig('./plt/hist1_{}.png'.format(psv))

# 看网络输出和gt
# gt = data[0].cpu().detach().numpy()
# et = data[1].cpu().detach().numpy()
# b_gt = data[2].cpu().detach().numpy()
# for B in range(32):
#     cv2.imwrite("./plt/gt_{}.png".format(B),gt[B,0])
#     cv2.imwrite("./plt/et_{}.png".format(B),et[B,0]*255)
#     cv2.imwrite("./plt/b_gt_{}.png".format(B),b_gt[B,0]*255)

# 代价体合并
# psv = data[0].cpu().detach().numpy()
# left = data[1].cpu().detach().numpy()
# right = data[2].cpu().detach().numpy()

# # for B in range(32):
# #     cv2.imwrite("./plt/psv_{}.png".format(B),psv[B,0].mean(0)*2000)
# #     cv2.imwrite("./plt/left_{}.png".format(B),left[B].mean(0)*2000)
# #     cv2.imwrite("./plt/right_{}.png".format(B),right[B].mean(0)*2000)

# a = psv[0,0]
# b = left[0]
# c = right[0]
# print(a = b)

# 看输入图
# def to_png(img):
#     img = (img*0.229+0.485)*255
#     img = np.clip(img,0,255).astype(int)
#     img = np.transpose(img, (1, 2, 0)) 
#     return img

# left = data[0].cpu().detach().numpy()
# right = data[1].cpu().detach().numpy()
# disp_l = data[2].cpu().detach().numpy()


# B = 0
# cv2.imwrite("./plt/left_{}.png".format(B),to_png(left))
# cv2.imwrite("./plt/right_{}.png".format(B),to_png(right))
# cv2.imwrite("./plt/disp_l_{}.png".format(B),disp_l)

# 看注意力指导图

# def to_png(img):
#     img = img*25500
#     img = np.clip(img,0,255).astype(int)
#     return img

# data = data.mean(dim=1).cpu().numpy()
# data = data[0]

# for d in range(data.shape[0]):
#     # cv2.imwrite("./plt/attention_{}.png".format(d),to_png(data[d]))
#     fig = plt.figure(figsize=(24, 6))
#     plt.hist(data[d].flatten(), bins=50)
#     plt.savefig('./plt/hist1_{}.png'.format(d))
#     plt.close()

# 看 warp图
def to_png(img):
    img = img*25500
    img = np.clip(img,0,255).astype(int)
    return img

x = data[0].cpu().numpy()
vgrid = data[1].cpu().numpy()
output = data[2].cpu().numpy()

for B in range(x.shape[0]):
    for d in range(x.shape[1]):
        cv2.imwrite("./plt/x_{}_{}.png".format(B,d),to_png(x[B,d]))
        # cv2.imwrite("./plt/vgrid_{}.png".format(B),to_png(data[B]))
        cv2.imwrite("./plt/warp_{}_{}.png".format(B,d),to_png(output[B,d]))

def visualize_tensor_grid(tensor, save_name, figsize=(20, 12), fontsize=6):
    """
    可视化一个 (H, W, 2) 的张量，每个格子显示两个值
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    H, W, _ = tensor.shape

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制格子边框
    for i in range(H + 1):
        ax.plot([0, W], [i, i], color="black", linewidth=0.5)
    for j in range(W + 1):
        ax.plot([j, j], [0, H], color="black", linewidth=0.5)

    # 在每个格子里写两个值
    for i in range(H):
        for j in range(W):
            v1, v2 = tensor[i, j]
            ax.text(j + 0.05, i + 0.35, f"{v1:.2f}", fontsize=fontsize, ha="left", va="center")
            ax.text(j + 0.05, i + 0.75, f"{v2:.2f}", fontsize=fontsize, ha="left", va="center")

    # 设置坐标系
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    plt.savefig(save_name)
    plt.close()

for B in range(data[0].shape[0]):
    visualize_tensor_grid(vgrid[B],'./plt/grid_{}.svg'.format(B))