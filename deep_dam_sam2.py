import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import AutoModel
import cv2
import time
import os
import tempfile
import gc # For garbage collection, to be thorough

try:
    from sam2.build_sam import build_sam2_video_predictor
    from dam import disable_torch_init
except ImportError as e:
    print(f"Error importing SAM2/DAM utilities: {e}")
    print("Please ensure 'sam2' library is installed and 'dam.py' is accessible.")
    print("For SAM2, you might need to clone its repository and set PYTHONPATH.")
    exit()

# --- Configuration ---
MAX_IMAGE_WIDTH = 640
MAX_IMAGE_HEIGHT = 480
NUM_POINTS_HORIZONTAL = 14
NUM_POINTS_VERTICAL = 12
MIN_MASK_AREA_THRESHOLD = 50

# IMPORTANT: Make sure these paths are correct relative to where image_analyzer.py is run,
# or use absolute paths.
# Consider making these configurable if the script might be run from different locations.
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"

DAM_MODEL_ID = "nvidia/DAM-3B-Self-Contained"
DAM_QUERY = "<image>\nDescribe the object or region highlighted by the mask in detail."

# Global model instances and device
sam2_predictor = None
dam_model_instance = None
device = None
models_initialized = False # Flag to track initialization

def initialize_models(cfg_path=SAM2_MODEL_CFG, ckpt_path=SAM2_CHECKPOINT, dam_id=DAM_MODEL_ID):
    global sam2_predictor, dam_model_instance, device, models_initialized
    if models_initialized:
        print("Models already initialized.")
        return

    print("Initializing models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Disabling torch init (if applicable)...")
    disable_torch_init() # Ensure this function exists and works as intended

    print(f"Loading SAM2 model with config: {cfg_path} and checkpoint: {ckpt_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"SAM2 model config not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")
    sam2_predictor = build_sam2_video_predictor(cfg_path, ckpt_path, device=device)
    print("SAM2 predictor loaded.")

    print(f"Loading DAM model: {dam_id}")
    dam_loaded_model = AutoModel.from_pretrained(
        dam_id, trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    prompt_modes = {"focal_prompt": "full+focal_crop"}
    dam_model_instance = dam_loaded_model.init_dam(
        conv_mode='v1', prompt_mode=prompt_modes.get("focal_prompt", "focal_prompt")
    )
    print("DAM model instance initialized.")
    models_initialized = True
    print("Models initialized successfully.")

def unload_models():
    global sam2_predictor, dam_model_instance, device, models_initialized
    if not models_initialized:
        print("Models are not initialized or already unloaded.")
        return

    print("Unloading models...")
    if dam_model_instance is not None:
        # For Hugging Face models, moving to CPU then deleting can help
        # dam_model_instance.cpu() # This might error if init_dam returns a non-nn.Module
        del dam_model_instance
        dam_model_instance = None
        print("DAM model unloaded.")

    if sam2_predictor is not None:
        # SAM2 predictor might not have an explicit .cpu() or .close()
        del sam2_predictor
        sam2_predictor = None
        print("SAM2 predictor unloaded.")

    if device and device.type == 'cuda':
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")

    # It's also good practice to try to explicitly collect garbage
    gc.collect()

    device = None # Reset device
    models_initialized = False
    print("Models and resources unloaded.")


def preprocess_image(image_path, max_width, max_height):
    # ... (your existing code, no changes needed here)
    print(f"Loading and preprocessing image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    original_width, original_height = img.size
    if original_width > max_width or original_height > max_height:
        print(f"Resizing {original_width}x{original_height} to fit {max_width}x{max_height}.")
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        print(f"Resized: {img.width}x{img.height}")
    else:
        print(f"Image {original_width}x{original_height} within limits.")
    return img

def generate_grid_points(image_width, image_height, num_x, num_y):
    # ... (your existing code, no changes needed here)
    points = []
    step_x = image_width / (num_x + 1)
    step_y = image_height / (num_y + 1)
    for i in range(1, num_x + 1):
        for j in range(1, num_y + 1):
            points.append((int(i * step_x), int(j * step_y)))
    print(f"Generated {len(points)} grid points.")
    return points

def apply_sam2_for_point(inference_state, point_xy, input_label=1):
    global sam2_predictor, device # Relies on global sam2_predictor and device
    if sam2_predictor is None: raise ValueError("SAM2 predictor not initialized. Call initialize_models() first.")
    if inference_state is None: raise ValueError("SAM2 inference_state not initialized.")

    points_np = np.array([[int(point_xy[0]), int(point_xy[1])]], dtype=np.float32)
    labels_np = np.array([input_label], dtype=np.int32)

    sam2_predictor.reset_state(inference_state)

    mask_np = np.array([], dtype=bool)

    with torch.no_grad():
        autocast_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=(device.type == "cuda")):
            _, _, out_mask_logits = sam2_predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=0, obj_id=1,
                points=points_np, labels=labels_np
            )

            if out_mask_logits is not None and len(out_mask_logits) > 0:
                single_mask_logits = out_mask_logits[0]
                if single_mask_logits.ndim > 2:
                    single_mask_logits = single_mask_logits.squeeze()
                mask_tensor = (single_mask_logits > 0.0).cpu()
                if mask_tensor.ndim != 2:
                    print(f"Warning: Mask tensor from SAM2 is not 2D after processing! Shape: {mask_tensor.shape}")
                    return np.array([], dtype=bool)
                mask_np = mask_tensor.numpy().astype(bool)
            else:
                print("SAM2 did not return mask logits for the point.")
    return mask_np

def get_description_for_mask(image_pil, mask_np, query):
    global dam_model_instance # Relies on global dam_model_instance
    if dam_model_instance is None: raise ValueError("DAM model not initialized. Call initialize_models() first.")

    if mask_np.ndim != 2:
        print(f"ERROR in get_description_for_mask: mask_np is not 2D! Shape: {mask_np.shape}")
        return "Error: Received non-2D mask for description."
    if not np.any(mask_np):
        return "No significant region found by SAM2 for this point."

    mask_pil = Image.fromarray(mask_np.astype(np.uint8) * 255)
    description = dam_model_instance.get_description(
        image_pil, mask_pil, query,
        temperature=0.2, top_p=0.5, num_beams=1, max_new_tokens=128
    )
    return description.strip()

def get_bounding_box_from_mask(mask_np):
    # ... (your existing code, no changes needed here)
    if mask_np.ndim != 2:
        print(f"ERROR in get_bounding_box_from_mask: mask_np is not 2D! Shape: {mask_np.shape}")
        return None
    if not np.any(mask_np): return None
    rows_indices = np.where(np.any(mask_np, axis=1))[0]
    cols_indices = np.where(np.any(mask_np, axis=0))[0]
    if rows_indices.size == 0 or cols_indices.size == 0: return None
    rmin, rmax = rows_indices[0], rows_indices[-1]
    cmin, cmax = cols_indices[0], cols_indices[-1]
    if rmax < rmin or cmax < cmin : return None
    return [int(cmin), int(rmin), int(cmax), int(rmax)]

def visualize_results(image_pil, results_data, output_path="analysis_visualization.png"):
    # ... (your existing code, no changes needed here)
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.4
    font_color_point = (0,0,255); font_color_box = (0,255,0); thickness = 1
    drawn_text_bboxes = []

    for i, item in enumerate(results_data):
        point_coords = item['point_of_interest']; box = item['bounding_box_xyxy']
        px, py = point_coords
        cv2.circle(img_cv, (px,py), 3, font_color_point, -1)
        if box:
            x1,y1,x2,y2 = box
            cv2.rectangle(img_cv, (x1,y1), (x2,y2), font_color_box, thickness)
            text_to_display = f"D{i+1}"
            (text_w, text_h), _ = cv2.getTextSize(text_to_display, font, font_scale, thickness)
            text_pos_x = x1; text_pos_y = y1 - 7
            if text_pos_y < text_h: text_pos_y = y2 + text_h + 7
            current_text_bbox = (text_pos_x, text_pos_y - text_h, text_pos_x + text_w, text_pos_y)
            attempts = 0; max_attempts = 5
            while any(not (current_text_bbox[2] < db[0] or current_text_bbox[0] > db[2] or \
                           current_text_bbox[3] < db[1] or current_text_bbox[1] > db[3]) \
                      for db in drawn_text_bboxes) and attempts < max_attempts:
                text_pos_y += (text_h+2) * (1 if attempts%2==0 else -1)
                if text_pos_y < text_h: text_pos_y = y2 + text_h + 7 + (attempts*5)
                if text_pos_y + text_h > img_cv.shape[0]: text_pos_y = y1 - 7 - (attempts*5)
                current_text_bbox = (text_pos_x, text_pos_y - text_h, text_pos_x + text_w, text_pos_y)
                attempts += 1
            if attempts < max_attempts:
                cv2.putText(img_cv, text_to_display, (text_pos_x, text_pos_y), font, font_scale, font_color_box, thickness)
                drawn_text_bboxes.append(current_text_bbox)
            else:
                cv2.putText(img_cv, text_to_display, (px+5, py-5), font, font_scale, font_color_box, thickness)
    cv2.imwrite(output_path, img_cv)
    print(f"Visualization saved to {output_path}")
    print("\nVisualization Key:")
    for i, item in enumerate(results_data):
        print(f"P{i+1} (Point): {item['point_of_interest']}")
        print(f"D{i+1} (Desc): {item['description'][:70]}...")


def analyze_image_deeply(image_path, auto_manage_models=True):
    """
    Analyzes an image deeply using SAM2 and DAM models.

    Args:
        image_path (str): Path to the image file.
        auto_manage_models (bool): If True, models will be initialized at the start
                                   of this function call and unloaded at the end.
                                   If False, it's assumed models are already initialized
                                   by calling initialize_models() externally, and will be
                                   unloaded by calling unload_models() externally.

    Returns:
        tuple: (list_of_results, PIL.Image object of processed image)
               Returns (None, None) if models are not initialized and auto_manage_models is False.
    """
    global sam2_predictor, dam_model_instance, models_initialized # Added models_initialized

    # Model Management
    if auto_manage_models:
        if not models_initialized: # Only init if not already done
            initialize_models()
        # else: models were initialized externally, use them
    elif not models_initialized:
        print("Error: Models not initialized. Call initialize_models() first or use auto_manage_models=True.")
        return None, None

    try:
        processed_image_pil = preprocess_image(image_path, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)
        img_width, img_height = processed_image_pil.size

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_image_filename = "0000.jpg" # SAM2 video predictor expects frame names
            temp_image_path = os.path.join(temp_dir, temp_image_filename)
            if processed_image_pil.mode != 'RGB':
                processed_image_pil.convert('RGB').save(temp_image_path, "JPEG")
            else:
                processed_image_pil.save(temp_image_path, "JPEG")
            print(f"Saved temp image for SAM2: {temp_image_path}")

            if sam2_predictor is None: raise RuntimeError("SAM2 predictor not initialized correctly.")

            print("Initializing SAM2 inference state (can be slow)...")
            start_init_time = time.time()
            # SAM2 video predictor init_state expects a path to a directory of frames
            sam_inference_state = sam2_predictor.init_state(video_path=temp_dir)
            print(f"SAM2 inference state init in {time.time()-start_init_time:.2f}s.")

            grid_points = generate_grid_points(img_width,img_height,NUM_POINTS_HORIZONTAL,NUM_POINTS_VERTICAL)
            overall_ignored_mask = np.zeros((img_height, img_width), dtype=bool)
            results = []
            total_points = len(grid_points)
            analysis_loop_start_time = time.time()

            for i, point_xy in enumerate(grid_points):
                px, py = int(point_xy[0]), int(point_xy[1])
                print(f"\nPoint {i+1}/{total_points}: ({px},{py})")
                if overall_ignored_mask[py, px]:
                    print("Point in ignored region. Skipping.")
                    continue

                point_proc_start_time = time.time()
                print("Applying SAM2...")
                current_mask_np = apply_sam2_for_point(sam_inference_state, point_xy)
                
                print(f"SAM2 raw mask_np shape: {current_mask_np.shape}, dtype: {current_mask_np.dtype}")
                if current_mask_np.ndim == 0:
                    print("SAM2 returned an empty array. Skipping point.")
                    continue
                if current_mask_np.ndim > 2:
                    print("Squeezing mask from SAM2...")
                    current_mask_np_squeezed = np.squeeze(current_mask_np)
                    print(f"SAM2 squeezed mask_np shape: {current_mask_np_squeezed.shape}")
                    if current_mask_np_squeezed.ndim == 0 and current_mask_np.size > 0 :
                         print("Warning: Squeeze resulted in 0-dim array from non-empty. Treating as no mask.")
                         current_mask_np = np.array([], dtype=bool)
                    else:
                        current_mask_np = current_mask_np_squeezed

                if current_mask_np.ndim != 2:
                    print(f"ERROR: Mask is not 2D after squeeze! Shape: {current_mask_np.shape}. Skipping point.")
                    continue

                if not np.any(current_mask_np) or current_mask_np.sum() < MIN_MASK_AREA_THRESHOLD:
                    print("SAM2 no significant mask.")
                    continue

                bounding_box = get_bounding_box_from_mask(current_mask_np)
                if bounding_box:
                    print(f"SAM2 mask. Box: {bounding_box}")
                else:
                    print("SAM2 mask, but no valid box. Area sum:", current_mask_np.sum(), "Shape:", current_mask_np.shape)
                    continue

                print("Getting DAM description...")
                description = get_description_for_mask(processed_image_pil, current_mask_np, DAM_QUERY)
                print(f"Desc: {description[:100]}...")
                results.append({"point":point_xy, "description":description, "box":bounding_box, "mask_sum":int(current_mask_np.sum())})
                overall_ignored_mask = np.logical_or(overall_ignored_mask, current_mask_np)
                print(f"Point proc time: {time.time()-point_proc_start_time:.2f}s. Ignored: {overall_ignored_mask.sum()/overall_ignored_mask.size*100:.2f}%")

            print(f"\n--- Analysis Loop Complete ---")
            print(f"Point proc loop time: {time.time()-analysis_loop_start_time:.2f}s. Found {len(results)} descs.")

        # Prepare output for visualization
        final_output_list_for_viz = []
        for res_item in results:
            final_output_list_for_viz.append({
                "description": res_item["description"],
                "point_of_interest": res_item["point"],
                "bounding_box_xyxy": res_item["box"]
            })
        return final_output_list_for_viz, processed_image_pil

    finally:
        if auto_manage_models:
            unload_models()


if __name__ == "__main__":
    # This block is for testing image_analyzer.py directly
    IMAGE_FILE_PATH_TEST = r"img\1.png" # CHANGE THIS
    # Or provide a generic placeholder and instruction
    # IMAGE_FILE_PATH_TEST = "path/to/your/test_image.png"
    # print(f"Testing with image: {IMAGE_FILE_PATH_TEST}")
    # print("Ensure SAM2_MODEL_CFG and SAM2_CHECKPOINT paths are correct in this script.")

    if "path/to/your" in SAM2_MODEL_CFG or "path/to/your" in SAM2_CHECKPOINT:
        print("ERROR: Please update SAM2_MODEL_CFG and SAM2_CHECKPOINT paths in image_analyzer.py before running.")
        exit()
    
    if not os.path.exists(IMAGE_FILE_PATH_TEST):
        print(f"ERROR: Test image not found at {IMAGE_FILE_PATH_TEST}")
        print("Please update IMAGE_FILE_PATH_TEST in the __main__ block of image_analyzer.py.")
        exit()

    overall_start_time = time.time()
    try:
        # Using auto_manage_models=True for standalone test
        analysis_results, final_image_pil = analyze_image_deeply(IMAGE_FILE_PATH_TEST, auto_manage_models=True)

        if analysis_results is not None: # Check if analysis was successful
            print("\n--- Final Results List ---")
            for i, item in enumerate(analysis_results):
                print(f"\nItem {i+1}: Pt: {item['point_of_interest']}, Box: {item['bounding_box_xyxy']}, Desc: {item['description']}")
            if final_image_pil:
                 visualize_results(final_image_pil, analysis_results, "detailed_analysis_output_standalone.png")
        else:
            print("Analysis did not complete successfully.")

    except FileNotFoundError as e: print(f"ERROR: File not found: {e}")
    except ImportError as e: print(f"ERROR: Import failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # If auto_manage_models=True, unload_models() is called within analyze_image_deeply.
        # If you were managing models externally, you'd call unload_models() here.
        print(f"\nTotal script exec time: {time.time()-overall_start_time:.2f}s")