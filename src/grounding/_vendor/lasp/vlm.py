"""LaSP VLM evaluation used by the extended CLI; arithmetic and prompt retained."""

import os
import json
from json import JSONDecodeError
import cv2
import numpy as np
from PIL import Image

from .vlm_utils import resize_image_to_GPT_size, encode_PIL_image_to_base64, user_prompt


def query_vlm(scan_id, caption, filtered_candidates, *, client, data_root):
    image_root = data_root / 'frames'
    masks_path = (data_root / 'nr3d_masks') / scan_id
    image_dirs = os.listdir(masks_path)
    base64Frames = []
    merged_indices = {}
    for obj_name in image_dirs:
        obj_id = int(obj_name.split("_")[0])
        if obj_id not in filtered_candidates:
            continue
        indices = np.load(masks_path / obj_name / "indices.npz")
        for k in indices.keys():
            img_name = k
            if img_name not in merged_indices:
                merged_indices[img_name] = {}
            merged_indices[img_name][obj_id] = indices[k]

    merged_areas = {}
    for img_name in merged_indices.keys():
        img = cv2.imread(str(image_root / scan_id / "color" / img_name))
        area = 0
        for obj_id in merged_indices[img_name].keys():
            indices = merged_indices[img_name][obj_id]
            area += (indices[1].max() - indices[1].min()) * (indices[0].max() - indices[0].min())
        merged_areas[img_name] = area

    # top 8 images on area
    sorted_images = sorted(merged_areas.items(), key=lambda x: x[1], reverse=True)
    selected_candidates = []

    top8_images = sorted_images[:8]
    for img_name, area in top8_images:
        for obj_id in merged_indices[img_name].keys():
            selected_candidates.append(obj_id)
    # if not all objects is selected, add to the top 8 images
    length = 8
    for obj_id in filtered_candidates:
        if obj_id not in selected_candidates:
            for sorted_img_name, _ in sorted_images:
                if obj_id in merged_indices[sorted_img_name]:
                    top8_images.append((sorted_img_name, merged_areas[sorted_img_name]))
                    length -= 1
                    break
    if length != 8:
        top8_images = top8_images[:length] + top8_images[-(8 - length):]
    assert len(top8_images) <= 8
    single_img_size = img.shape[:2]
    stitched_img = np.ones((single_img_size[0] * 2, single_img_size[1] * 4, 3), dtype=np.uint8) * 255
    for i in range(2):
        for j in range(4):
            if i * 4 + j >= len(top8_images):
                continue
            img_name, area = top8_images[i * 4 + j]
            img = cv2.imread(str(image_root / scan_id / "color" / img_name))
            for obj_id, area in merged_indices[img_name].items():
                indices = merged_indices[img_name][obj_id]
                cv2.putText(img, f"obj_{obj_id}", (int((indices[1].max() + indices[1].min()) / 2) , int((indices[0].max() + indices[0].min()) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            stitched_img[i * single_img_size[0]: (i + 1) * single_img_size[0], j * single_img_size[1]: (j + 1) * single_img_size[1]] = img
    stitched_img = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
    # Imported here, not at module scope: this is the only use, and a top-level
    # import made every non-VLM eval depend on scikit-image.
    from skimage import img_as_ubyte
    image = Image.fromarray(img_as_ubyte(stitched_img))
    image = resize_image_to_GPT_size(image)

    ecd = encode_PIL_image_to_base64(image)
    base64Frames.append(ecd)
    assert len(base64Frames) == 1

    messages = [
        {
            "role": "system",
            "content": "You are good at finding objects specified by a description in indoor rooms by watching the videos scanning the rooms."
        },
        {
            "role": "user",
            "content": None,
        }
    ]
    messages[1]["content"] = [
        {
            "type": "text",
            "text": user_prompt.format(utterance=caption, candidates=str(filtered_candidates))
        },
        *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}", "detail": "high"}}, base64Frames),
    ]
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": messages,
        "top_p": 0.3,
        "temperature": 0.1,
    }

    openai_response = client.chat.completions.create(
        **payload
    ).choices[0].message.content

    try:
        if "```json" in openai_response:
            openai_response = openai_response.split("```json")[1].split("```")[0].strip()
        json_obj = json.loads(openai_response)
        return int(json_obj['object id'])
    except (JSONDecodeError, KeyError):
        return -1

