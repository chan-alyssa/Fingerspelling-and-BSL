from pathlib import Path
import os.path as osp
import h5py
import numpy as np
import argparse
import tarfile
import re
import os
import cv2
from hamer.utils.renderer import Renderer, cam_crop_to_full
from library.readh5 import ReadH5
from PIL import Image
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
import torch
import io
from hamer.models import MANO

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--vid', type=str, help='Video ID to run inference on')
    parser.add_argument('--img_folder', type=str, default='demo2', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='epicdemo', help='Output folder to save rendered results')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    args = parser.parse_args()


    # Setup the renderer
    # model_cfg ={}
    # model_cfg.EXTRA.FOCAL_LENGTH = 5000
    # model_cfg.MODEL.IMAGE_SIZE = 224
    # model_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
    # model_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
    model, model_cfg = load_hamer(args.checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    

    mano_cfg = {'data_dir': '_DATA/data/', 'model_path':'_DATA/data/mano/',
                'gender':'neutral','num_hand_joints':15,'mean_params':'./_DATA/data/mano_mean_params.npz','create_body_pose': False}
    mano = MANO(**mano_cfg)


    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    vid = args.vid

    # Filter the list
    file_list = [os.path.join(args.img_folder, file) for file in os.listdir(args.img_folder) if file.endswith('.png')]
    file_list.sort()

    # hf = h5py.File(f'{args.data_folder}/hamer_out/{vid}.h5', 'r')
    # data1 = hf[f'{vid}/hand_pose']

    read_h5file = ReadH5('images.h5')

    # data1=data.read_sequence(f'{vid}/hand_pose') #286

    # print(data1)

    # 0 to 100 frames

    frame_len = 10


    for ii in range(frame_len):
        all_verts = []
        all_cam_t = []
        all_right = []        
        #image ii load

        # if data1 left hand exist, parse left hand
        # convert left to vertice

        # if data1 right hand exist, parse right hand

        # Add all verts and cams to list

        img_path = file_list[ii]
        # frame_int = int(img_fn.split('_')[-1])  # Split by underscore and get the last part, then convert to integer
        # print("frame_int", frame_int)
        # data1[frame_int, 0] = frame_int

        img_cv2 = cv2.imread(str(img_path))
        
        data = read_h5file.read_sequence_slice(f'{vid}/hand_pose', ii)
        # data = data1()

        frame_num = int(data[0])
        left_exist = int(data[1])
        right_exist = int(data[162])
        
        #print(data)

        if not right_exist:
            continue

        arr_left = data[2:162]
        
        box_center_left = torch.tensor([arr_left[:2]])
        box_size_left = torch.tensor([arr_left[2]])
        pred_cam_left = torch.tensor([arr_left[3:6]])
        global_orient_left = torch.tensor([np.reshape(arr_left[6:15], (1,3,3))])
        hand_pose_left = torch.tensor([np.reshape(arr_left[15:150],(15,3,3))])
        betas_left = torch.tensor([arr_left[150:]])

        arr_right = data[163:]
        box_center_right = torch.tensor([arr_right[:2]])
        box_size_right = torch.tensor([arr_right[2]])
        pred_cam_right = torch.tensor([arr_right[3:6]])
        global_orient_right = torch.tensor([np.reshape(arr_right[6:15], (1,3,3))])
        hand_pose_right = torch.tensor([np.reshape(arr_right[15:150],(15,3,3))])
        betas_right = torch.tensor([arr_right[150:]])

        # box_center, box_size, pred_cam_cpu, global_orient,hand_pose,betas
        # print("pred_cam_cpu[n]",np.shape(pred_cam_cpu[n])) # 3
        # print("global_orient[n]",np.shape(global_orient[n])) # 1,3,3
        # print("hand_pose[n]",np.shape(hand_pose[n])) # 15,3,3
        # print("betas[n]",np.shape(betas[n]))   # 10

        # Compute model vertices, joints and the projected joints
        pred_mano_params={'global_orient':global_orient_right,'hand_pose':hand_pose_right,
                          'betas':betas_right}
        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(1, -1, 3, 3)
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(1,-1, 3, 3)
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(1,-1)
        # print("pred_mano_params",pred_mano_params)
        mano_output = mano(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices.numpy()
        pred_keypoints_3d= pred_keypoints_3d.reshape(-1, 3)
        pred_vertices = pred_vertices.reshape(-1, 3)

        img_size = torch.tensor([[np.shape(img_cv2)[1],np.shape(img_cv2)[0]]]).float()
        # print("img_size",img_size)
        # print("img_size.max()",img_size.max())
        FOCAL_LENGTH = 5000
        MODEL_IMG_SIZE = 224
        scaled_focal_length = FOCAL_LENGTH / MODEL_IMG_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam_right, box_center_right, box_size_right, img_size, scaled_focal_length).detach().cpu().numpy()
        verts = pred_vertices
        
        is_right = 1 #left
        verts[:,0] = (2*is_right-1)*verts[:,0]
        cam_t = pred_cam_t_full
        all_verts.append(verts)
        all_cam_t.append(cam_t)
        all_right.append(is_right)
        
        misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
                )
        cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[0], is_right=all_right, **misc_args)
    
        # Overlay image
        input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
        input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

        output_path = os.path.join(args.out_folder, f'{frame_num}_all.jpg')
        print(f'{output_path}')
        cv2.imwrite(output_path, (255 * input_img_overlay[:, :, ::-1]))

    # g1 = hf.create_group(f'{vid}')
    # file_len = len(file_list) #As the frame starts from 1
    # d1 = np.zeros((file_len,316 + 7))
    # g1.create_dataset('hand_pose',data=d1)

    # data1 = hf[f'{vid}/hand_pose']



if __name__ == '__main__':
    main()