import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath,select=(0,1,2)):
    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_finalpass') > -1] #找到所有包含原始图片的文件夹
    disp = [dsp for dsp in classes if dsp.find('disparity') > -1]  #找到所有包含视差文件的文件夹

    # monkaa_part
    if 0 in select:
        monkaa_path = filepath + [x for x in image if 'monkaa' in x][0] #找到monkaa原始图片的文件夹
        monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0] #找到monkaa视差文件的文件夹
        monkaa_path = monkaa_path + '/frames_finalpass/'
        monkaa_disp = monkaa_disp + '/disparity/'

        monkaa_dir = os.listdir(monkaa_path)
        for dd in monkaa_dir:
            for im in os.listdir(monkaa_path + '/' + dd + '/left/'):
                if is_image_file(monkaa_path + '/' + dd + '/left/' + im):
                    all_left_img.append(monkaa_path + '/' + dd + '/left/' + im) #对每个列出文件判断，如果是图片文件，则添加到列表中
                    all_left_disp.append(monkaa_disp + '/' + dd + '/left/' + im.split(".")[0] + '.pfm')  #对每个列出文件判断，如果是视差文件，则添加到此列表中

            for im in os.listdir(monkaa_path + '/' + dd + '/right/'):
                if is_image_file(monkaa_path + '/' + dd + '/right/' + im):
                    all_right_img.append(monkaa_path + '/' + dd + '/right/' + im) #对右侧图像做同处理。右侧图片不含视差文件

    if 1 in select:
        flying_path = filepath + [x for x in image if x == 'frames_finalpass'][0] #找到包含飞行图像的文件夹，这个文件夹名字就是frames_finalpass
        flying_disp = filepath + [x for x in disp if x == 'disparity'][0] #找到包含飞行图像的视差文件的文件夹
        flying_dir = flying_path + '/TRAIN/' #飞行图像的训练集目录
        subdir = ['A', 'B', 'C'] #飞行图像的子目录

        for ss in subdir:
            flying = os.listdir(flying_dir + ss)

            for ff in flying:
                imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
                for im in imm_l:
                    if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                        all_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im) #添加左图

                    all_left_disp.append(flying_disp + '/TRAIN/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm') #添加左图对应的视差文件，在另一个文件夹中

                    if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im): #添加右图
                        all_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

        flying_dir = flying_path + '/TEST/' #飞行图像的测试集目录

        subdir = ['A', 'B', 'C']

        for ss in subdir:
            flying = os.listdir(flying_dir + ss)

            for ff in flying:
                imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
                for im in imm_l:
                    if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                        test_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im)

                    test_left_disp.append(flying_disp + '/TEST/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm')

                    if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im):
                        test_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

    if 2 in select:
        driving_dir = filepath + [x for x in image if 'driving' in x][0]
        driving_disp = filepath + [x for x in disp if 'driving' in x][0]

        driving_dir = driving_dir + '/frames_finalpass/'
        driving_disp = driving_disp + '/disparity/'

        subdir1 = ['35mm_focallength', '15mm_focallength'] #一级子目录
        subdir2 = ['scene_backwards', 'scene_forwards'] #二级子目录
        subdir3 = ['fast', 'slow'] #三级子目录

        for i in subdir1:
            for j in subdir2:
                for k in subdir3:
                    imm_l = os.listdir(driving_dir + i + '/' + j + '/' + k + '/left/')
                    for im in imm_l:
                        if is_image_file(driving_dir + i + '/' + j + '/' + k + '/left/' + im):
                            all_left_img.append(driving_dir + i + '/' + j + '/' + k + '/left/' + im)
                        all_left_disp.append(
                            driving_disp + '/' + i + '/' + j + '/' + k + '/left/' + im.split(".")[0] + '.pfm')

                        if is_image_file(driving_dir + i + '/' + j + '/' + k + '/right/' + im):
                            all_right_img.append(driving_dir + i + '/' + j + '/' + k + '/right/' + im)
        #基本上和上面一样的逻辑，添加左图、右图和视差文件

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


# def dataloader(filepath):
#     filepath += '/'
#     classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
#     image = [img for img in classes if img.find('frames_cleanpass') > -1]
#     disp = [dsp for dsp in classes if dsp.find('disparity') > -1]
    
#     monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
#     monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]

#     monkaa_dir = os.listdir(monkaa_path)

#     all_left_img = []
#     all_right_img = []
#     all_left_disp = []
#     test_left_img = []
#     test_right_img = []
#     test_left_disp = []

#     for dd in monkaa_dir:
#         for im in os.listdir(monkaa_path + '/' + dd + '/left/'):
#             if is_image_file(monkaa_path + '/' + dd + '/left/' + im):
#                 all_left_img.append(monkaa_path + '/' + dd + '/left/' + im)
#                 all_left_disp.append(monkaa_disp + '/' + dd + '/left/' + im.split(".")[0] + '.pfm')

#         for im in os.listdir(monkaa_path + '/' + dd + '/right/'):
#             if is_image_file(monkaa_path + '/' + dd + '/right/' + im):
#                 all_right_img.append(monkaa_path + '/' + dd + '/right/' + im)

#     flying_path = filepath + [x for x in image if x == 'frames_cleanpass'][0]
#     flying_disp = filepath + [x for x in disp if x == 'frames_disparity'][0]
#     flying_dir = flying_path + '/TRAIN/'
#     subdir = ['A', 'B', 'C']

#     for ss in subdir:
#         flying = os.listdir(flying_dir + ss)

#         for ff in flying:
#             imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
#             for im in imm_l:
#                 if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
#                     all_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im)

#                 all_left_disp.append(flying_disp + '/TRAIN/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm')

#                 if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im):
#                     all_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

#     flying_dir = flying_path + '/TEST/'

#     subdir = ['A', 'B', 'C']

#     for ss in subdir:
#         flying = os.listdir(flying_dir + ss)

#         for ff in flying:
#             imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
#             for im in imm_l:
#                 if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
#                     test_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im)

#                 test_left_disp.append(flying_disp + '/TEST/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm')

#                 if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im):
#                     test_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

#     driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
#     driving_disp = filepath + [x for x in disp if 'driving' in x][0]

#     subdir1 = ['15mm_focallength', '15mm_focallength']
#     subdir2 = ['scene_backwards', 'scene_forwards']
#     subdir3 = ['fast', 'slow']

#     for i in subdir1:
#         for j in subdir2:
#             for k in subdir3:
#                 imm_l = os.listdir(driving_dir + i + '/' + j + '/' + k + '/left/')
#                 for im in imm_l:
#                     if is_image_file(driving_dir + i + '/' + j + '/' + k + '/left/' + im):
#                         all_left_img.append(driving_dir + i + '/' + j + '/' + k + '/left/' + im)
#                     all_left_disp.append(
#                         driving_disp + '/' + i + '/' + j + '/' + k + '/left/' + im.split(".")[0] + '.pfm')

#                     if is_image_file(driving_dir + i + '/' + j + '/' + k + '/right/' + im):
#                         all_right_img.append(driving_dir + i + '/' + j + '/' + k + '/right/' + im)

#     return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp
