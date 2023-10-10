import numpy as np
import torch

from mmhuman3d.utils.demo_utils import box2cs, xywh2xyxy, xyxy2xywh
from mmhuman3d.apis.inference import LoadImage, Compose, collate, _indexing_sequence

def cliff_inference_image_based_model(
    model,
    img_or_path,
    det_results,
    bbox_thr=None,
    format='xywh',
):
    """Inference a single image with a list of person bounding boxes.

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (Union[str, np.ndarray]): Image filename or loaded image.
        det_results (List(dict)): the item in the dict may contain
            'bbox' and/or 'track_id'.
            'bbox' (4, ) or (5, ): The person bounding box, which contains
            4 box coordinates (and score).
            'track_id' (int): The unique id for each human instance.
        bbox_thr (float, optional): Threshold for bounding boxes.
            Only bboxes with higher scores will be fed into the pose detector.
            If bbox_thr is None, ignore it. Defaults to None.
        format (str, optional): bbox format ('xyxy' | 'xywh'). Default: 'xywh'.
            'xyxy' means (left, top, right, bottom),
            'xywh' means (left, top, width, height).

    Returns:
        list[dict]: Each item in the list is a dictionary,
            containing the bbox: (left, top, right, bottom, [score]),
            SMPL parameters, vertices, kp3d, and camera.
    """
    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']
    mesh_results = []
    if len(det_results) == 0:
        return []

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in det_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        det_results = [det_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return []

    cfg = model.cfg
    device = next(model.parameters()).device

    # build the data pipeline
    inference_pipeline = [LoadImage()] + cfg.inference_pipeline
    inference_pipeline = Compose(inference_pipeline)

    assert len(bboxes[0]) in [4, 5]

    batch_data = []
    input_size = cfg['img_resolution']
    aspect_ratio = 1 if isinstance(input_size,
                                   int) else input_size[0] / input_size[1]

    # Estimate focal length
    focal_len = torch.tensor(np.sqrt(np.square(input_size[0]) + np.square(input_size[1])))

    for i, bbox in enumerate(bboxes_xywh):
        center, scale = box2cs(bbox, aspect_ratio, bbox_scale_factor=1.25)
        # prepare data
        data = {
            'image_path': img_or_path,
            'center': center,
            'scale': scale,
            'rotation': 0,
            'bbox_score': bbox[4] if len(bbox) == 5 else 1,
            'sample_idx': i,
        }
        data = inference_pipeline(data)

        # cx, cy, crop_size,    all / focal_len
        data['bbox_info'] = torch.tensor(
            [center[0], center[1], scale.mean()],
            dtype=torch.float32) / focal_len
        data['img_h'] = input_size[0]
        data['img_w'] = input_size[1]
        data['center'] = data['img_metas'].data['center']
        data['scale'] = data['img_metas'].data['scale']

        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter not work so just move image to cuda device
        batch_data['img'] = batch_data['img'].to(device)
        batch_data['bbox_info'] = batch_data['bbox_info'].to(device)
        batch_data['img_h'] = batch_data['img_h'].to(device).unsqueeze(1)
        batch_data['img_w'] = batch_data['img_w'].to(device).unsqueeze(1)
        batch_data['center'] = batch_data['center'].to(device)
        batch_data['scale'] = batch_data['scale'].to(device)

    # get all img_metas of each bounding box
    batch_data['img_metas'] = [
        img_metas[0] for img_metas in batch_data['img_metas'].data
    ]

    focal_len = torch.tensor([focal_len] * len(batch_data['img_metas'])).to(device).unsqueeze(1)

    # forward the model
    with torch.no_grad():
        results = model(
            img=batch_data['img'],
            img_metas=batch_data['img_metas'],
            sample_idx=batch_data['sample_idx'],
            # kwgs
            bbox_info=batch_data['bbox_info'],
            img_h=batch_data['img_h'],
            img_w=batch_data['img_w'],
            center=batch_data['center'],
            scale=batch_data['scale'],
            focal_length=focal_len,
        )

    for idx in range(len(det_results)):
        mesh_result = det_results[idx].copy()
        mesh_result['bbox'] = bboxes_xyxy[idx]
        for key, value in results.items():
            mesh_result[key] = _indexing_sequence(value, index=idx)
        mesh_results.append(mesh_result)
    return mesh_results