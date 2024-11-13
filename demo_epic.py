from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from PIL import Image
import io
import re
import tarfile
import h5py
import os.path as osp

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--vid', type=str, help='Video ID to run inference on')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='demo2', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demoout2', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='regnety', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--viz', dest='viz', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--vid_write', dest='vid_write', action='store_true', default=False, help='If set, vid_write')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    if args.viz and args.vid_write:
        width = 456
        height = 256
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video=cv2.VideoWriter(f'{args.out_folder}/{args.vid}.mp4',fourcc,30,(width,height))

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)
    vid = args.vid

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]


    # Filter the list
    file_list = [os.path.join(args.img_folder, file) for file in os.listdir(args.img_folder) if file.endswith('.png')]
    file_list.sort()

    hf = h5py.File('images.h5', 'w')
    g1 = hf.create_group(f'{vid}')
    file_len = len(file_list) #As the frame starts from 1
    d1 = np.zeros((file_len,316 + 7))
    
    # hf[f'{vid}/hand_pose'] = data1
    # print("data1",data1[3])
    g1.create_dataset('hand_pose',data=d1)
    data1 = hf[f'{vid}/hand_pose']

    # Iterate over all images in folder
    cnt = 0
    for img_path in img_paths:
        cnt+=1
        # if cnt > 1000:
        #     continue
        # if cnt < 759:
        #     continue
        
        img_fn, _ = os.path.splitext(os.path.basename(img_path))  # Extract filename without extension
        # frame_int = int(img_fn.split('_')[-1])  # Split by underscore and get the last part, then convert to integer
        # print("frame_int", frame_int)
        data1[cnt, 0] = cnt

        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # print("pred_bboxes",pred_bboxes)

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # print("vitposes_out",vitposes_out)
        # print("pred_bboxes",np.shape(pred_bboxes))
        # print("vitposes_out",np.shape(vitposes_out))
        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            if sum(left_hand_keyp[:,2] > 0.7) > 2 and sum(right_hand_keyp[:,2] > 0.7) > 2:
                keyp = left_hand_keyp
                valid = keyp[:,2] > 0.5
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)

                keyp = right_hand_keyp
                valid = keyp[:,2] > 0.5
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)
            else:
                if sum(left_hand_keyp[:,2]) > sum(right_hand_keyp[:,2]):
                    # Rejecting not confident detections
                    keyp = left_hand_keyp
                    valid = keyp[:,2] > 0.5
                    if sum(valid) > 3:
                        bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                        bboxes.append(bbox)
                        is_right.append(0)
                else:
                    keyp = right_hand_keyp
                    valid = keyp[:,2] > 0.5
                    if sum(valid) > 3:
                        bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                        bboxes.append(bbox)
                        is_right.append(1)

            # print("left_hand_keyp",left_hand_keyp[:,2])
            # print("right_hand_keyp",right_hand_keyp[:,2])

        # if len(bboxes) == 0:
        #     continue
        if len(bboxes) == 0:
            if args.viz:
                video.write(img_cv2)
            continue
        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            
            # print("box_center",np.shape(box_center)) # 1,2
            # print("box_size",np.shape(box_size)) # 1
            
            img_size = batch["img_size"].float()
            # print("img_size",img_size)
            # print("img_size",np.shape(img_size))
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            pred_cam_cpu = out['pred_cam'].cpu()
            pred_mano_params = out['pred_mano_params']
            global_orient = pred_mano_params['global_orient'].cpu()
            hand_pose = pred_mano_params['hand_pose'].cpu()
            betas = pred_mano_params['betas'].cpu()

            
            # print("batch",batch.keys())

            # Render the result
            batch_size = batch['img'].shape[0]
            if batch_size > 1: # If they predict more than one, save the first one.
                if batch['right'][0]:
                    data1[cnt,158+4] = 1
                    arr = np.concatenate(( box_center[0].cpu().numpy().flatten(), box_size[0].cpu().numpy().flatten(), pred_cam_cpu[0].numpy().flatten(), global_orient[0].numpy().flatten(),hand_pose[0].numpy().flatten(),betas[0].numpy().flatten()), axis=0)
                    data1[cnt,159+4:] = arr
                else:
                    data1[cnt,158+4] = 1
                    arr = np.concatenate(( box_center[0].cpu().numpy().flatten(), box_size[0].cpu().numpy().flatten(), pred_cam_cpu[0].numpy().flatten(), global_orient[0].numpy().flatten(),hand_pose[0].numpy().flatten(),betas[0].numpy().flatten()), axis=0)
                    data1[cnt,2:158+4] = arr            
            else:
                if batch['right']:
                    data1[cnt,158+4] = 1
                    arr = np.concatenate(( box_center.cpu().numpy().flatten(), box_size.cpu().numpy().flatten(), pred_cam_cpu.numpy().flatten(), global_orient.numpy().flatten(),hand_pose.numpy().flatten(),betas.numpy().flatten()), axis=0)
                    data1[cnt,159+4:] = arr
                else:
                    data1[cnt,1] = 1
                    arr = np.concatenate(( box_center.cpu().numpy().flatten(), box_size.cpu().numpy().flatten(), pred_cam_cpu.numpy().flatten(), global_orient.numpy().flatten(),hand_pose.numpy().flatten(),betas.numpy().flatten()), axis=0)
                    data1[cnt,2:158+4] = arr

            if args.viz:
                for n in range(batch_size):
                    # Get filename from path img_path
                    person_id = int(batch['personid'][n])
                    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                    input_patch = input_patch.permute(1,2,0).numpy()

                    regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            batch['img'][n],
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            )

                    if args.side_view:
                        side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                                white_img,
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(1, 1, 1),
                                                side_view=True)
                        final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                    else:
                        final_img = np.concatenate([input_patch, regression_img], axis=1)

                    if not args.vid_write:
                        cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                    # Add all verts and cams to list
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    is_right = batch['right'][n].cpu().numpy()
                    verts[:,0] = (2*is_right-1)*verts[:,0]

                    # print("pred_vertices",verts)
                    # print("pred_vertices",np.shape(verts))
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

                    # Save all meshes to disk
                    if args.save_mesh:
                        camera_translation = cam_t.copy()
                        tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                        tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        if args.viz:
            # Render front view
            if args.full_frame and len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

                # Overlay image
                input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                
                
                if args.vid_write:
                    overlayed_img = 255*input_img_overlay[:, :, ::-1]
                    video.write(overlayed_img.astype(np.uint8))
                else:
                    cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])
    if args.vid_write:
        cv2.destroyAllWindows()
        video.release()
    hf.close()
if __name__ == '__main__':
    main()