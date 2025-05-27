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
    from dam import disable_torch_init # Assuming dam.py is in the same directory or PYTHONPATH
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

SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"

DAM_MODEL_ID = "nvidia/DAM-3B-Self-Contained"
DAM_QUERY = "<image>\nDescribe the object or region highlighted by the mask in detail."

# Output Folder Configuration
OUTPUT_BASE_FOLDER_V3 = "analysis_output_standalone_v3"
OUTPUT_TEXT_FOLDER_V3 = os.path.join(OUTPUT_BASE_FOLDER_V3, "texts")
OUTPUT_IMAGE_FOLDER_V3 = os.path.join(OUTPUT_BASE_FOLDER_V3, "images")


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
    disable_torch_init()

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
        del dam_model_instance
        dam_model_instance = None
        print("DAM model unloaded.")

    if sam2_predictor is not None:
        del sam2_predictor
        sam2_predictor = None
        print("SAM2 predictor unloaded.")

    if device and device.type == 'cuda':
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    gc.collect()
    device = None
    models_initialized = False
    print("Models and resources unloaded.")


def preprocess_image(image_path, max_width, max_height):
    print(f"Loading and preprocessing image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    original_width, original_height = img.size
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"Invalid image dimensions: {original_width}x{original_height} for {image_path}")

    if original_width > max_width or original_height > max_height:
        print(f"Resizing {original_width}x{original_height} to fit within {max_width}x{max_height}.")
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        print(f"Resized to: {img.width}x{img.height}")
    else:
        print(f"Image {original_width}x{original_height} is within limits. No resize needed.")
    return img

def generate_grid_points(image_width, image_height, num_x, num_y):
    points = []
    if image_width <= 0 or image_height <=0 or num_x <= 0 or num_y <= 0:
        print("Warning: Cannot generate grid points with non-positive dimensions or counts.")
        return points

    step_x = image_width / (num_x + 1)
    step_y = image_height / (num_y + 1)
    for i in range(1, num_x + 1):
        for j in range(1, num_y + 1):
            px = int(i * step_x)
            py = int(j * step_y)
            px = max(0, min(px, image_width - 1))
            py = max(0, min(py, image_height - 1))
            points.append((px, py))
    print(f"Generated {len(points)} grid points.")
    return points

def apply_sam2_for_point(inference_state, point_xy, input_label=1):
    global sam2_predictor, device
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
                    if single_mask_logits.shape[0] == 1: # Squeeze if singleton first dim
                        single_mask_logits = single_mask_logits.squeeze(0)
                    else: # Otherwise, take the first mask (e.g., if multiple ambiguous masks)
                        single_mask_logits = single_mask_logits[0]

                mask_tensor = (single_mask_logits > 0.0).cpu()
                if mask_tensor.ndim != 2:
                    print(f"Warning: Mask tensor from SAM2 is not 2D after processing! Shape: {mask_tensor.shape}")
                    return np.array([], dtype=bool)
                mask_np = mask_tensor.numpy().astype(bool)
            else:
                print("SAM2 did not return mask logits for the point.")
    return mask_np

def get_description_for_mask(image_pil, mask_np, query):
    global dam_model_instance
    if dam_model_instance is None: raise ValueError("DAM model not initialized. Call initialize_models() first.")

    if mask_np.ndim != 2:
        print(f"ERROR in get_description_for_mask: mask_np is not 2D! Shape: {mask_np.shape}")
        return "Error: Received non-2D mask for description."
    if not np.any(mask_np):
        return "No significant region found by SAM2 for this point."

    mask_pil = Image.fromarray(mask_np.astype(np.uint8) * 255)
    # Ensure image_pil is RGB for DAM
    if image_pil.mode != 'RGB':
        image_pil_rgb = image_pil.convert('RGB')
    else:
        image_pil_rgb = image_pil

    description = "Error: DAM description generation failed."
    try:
        description = dam_model_instance.get_description(
            image_pil_rgb, mask_pil, query,
            temperature=0.2, top_p=0.5, num_beams=1, max_new_tokens=128
        )
        description = description.strip()
    except Exception as e:
        print(f"Error during DAM description generation: {e}")
    return description


def get_bounding_box_from_mask(mask_np):
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

def visualize_results(image_pil, results_data, output_path):
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
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
    cv2.imwrite(output_path, img_cv)
    print(f"Visualization saved to {output_path}")
    print("\nVisualization Key (also in text report):")
    for i, item in enumerate(results_data):
        print(f"P{i+1} (Point): {item['point_of_interest']}")
        print(f"D{i+1} (Desc): {item['description'][:70]}...")

def analyze_image_deeply(image_path, auto_manage_models=True):
    global sam2_predictor, dam_model_instance, models_initialized

    if auto_manage_models:
        if not models_initialized:
            initialize_models()
    elif not models_initialized:
        print("Error: Models not initialized. Call initialize_models() first or use auto_manage_models=True.")
        return None, None # Return None, None to indicate failure

    try:
        processed_image_pil = preprocess_image(image_path, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)
        img_width, img_height = processed_image_pil.size

        if img_width <= 0 or img_height <= 0:
            print(f"ERROR: Processed image has invalid dimensions: {img_width}x{img_height}. Aborting analysis.")
            # Return empty results but the (potentially invalid) image for consistency if needed by caller
            return [], processed_image_pil

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_image_filename = "0000.jpg"
            temp_image_path = os.path.join(temp_dir, temp_image_filename)
            processed_image_pil.save(temp_image_path, "JPEG")
            print(f"Saved temp image for SAM2: {temp_image_path}")

            if sam2_predictor is None: raise RuntimeError("SAM2 predictor not initialized correctly.")

            print("Initializing SAM2 inference state (can be slow)...")
            start_init_time = time.time()
            sam_inference_state = sam2_predictor.init_state(video_path=temp_dir)
            print(f"SAM2 inference state init in {time.time()-start_init_time:.2f}s.")

            grid_points = generate_grid_points(img_width,img_height,NUM_POINTS_HORIZONTAL,NUM_POINTS_VERTICAL)
            overall_ignored_mask = np.zeros((img_height, img_width), dtype=bool)
            results = []
            total_points = len(grid_points)
            analysis_loop_start_time = time.time()

            for i, point_xy_tuple in enumerate(grid_points):
                px, py = int(point_xy_tuple[0]), int(point_xy_tuple[1])
                print(f"\nPoint {i+1}/{total_points}: ({px},{py})")

                if not (0 <= px < img_width and 0 <= py < img_height):
                    print(f"ERROR: Point ({px},{py}) is out of image bounds ({img_width}x{img_height}). Skipping.")
                    continue
                if overall_ignored_mask[py, px]:
                    print("Point in ignored region. Skipping.")
                    continue

                point_proc_start_time = time.time()
                print("Applying SAM2...")
                current_mask_np = apply_sam2_for_point(sam_inference_state, point_xy_tuple)
                
                if current_mask_np.ndim == 0 or current_mask_np.size == 0:
                    print("SAM2 returned an empty array. Skipping point.")
                    continue
                if current_mask_np.ndim != 2: # apply_sam2_for_point should handle squeeze
                    print(f"ERROR: Mask is not 2D! Shape: {current_mask_np.shape}. Skipping point.")
                    continue

                # Ensure mask has same dimensions as image for consistency
                if current_mask_np.shape != (img_height, img_width):
                    print(f"Warning: SAM2 mask shape {current_mask_np.shape} differs from image H,W {(img_height, img_width)}. Attempting resize.")
                    try:
                        pil_mask_for_resize = Image.fromarray(current_mask_np.astype(np.uint8) * 255)
                        pil_mask_resized = pil_mask_for_resize.resize((img_width, img_height), Image.Resampling.NEAREST)
                        current_mask_np = np.array(pil_mask_resized).astype(bool)
                        print(f"Mask resized to {current_mask_np.shape}")
                    except Exception as e_resize:
                        print(f"Error resizing mask: {e_resize}. Skipping point.")
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
                results.append({"point":point_xy_tuple, "description":description, "box":bounding_box, "mask_sum":int(current_mask_np.sum())})
                overall_ignored_mask = np.logical_or(overall_ignored_mask, current_mask_np)
                print(f"Point proc time: {time.time()-point_proc_start_time:.2f}s. Ignored: {overall_ignored_mask.sum()/overall_ignored_mask.size*100:.2f}%")

            print(f"\n--- Analysis Loop Complete ---")
            print(f"Point proc loop time: {time.time()-analysis_loop_start_time:.2f}s. Found {len(results)} descs.")

        final_output_list_for_viz = []
        for res_item in results:
            final_output_list_for_viz.append({
                "description": res_item["description"],
                "point_of_interest": res_item["point"],
                "bounding_box_xyxy": res_item["box"]
                # No 'source' here as this version only uses grid points
            })
        return final_output_list_for_viz, processed_image_pil

    finally:
        if auto_manage_models:
            unload_models()

def save_analysis_text_v3(analysis_results, input_image_filename, text_output_path):
    """Saves the detailed analysis results to a text file for v3."""
    os.makedirs(os.path.dirname(text_output_path), exist_ok=True)
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write(f"Deep Analysis Report (Standalone V3) for Image: {os.path.basename(input_image_filename)}\n")
        f.write(f"Models Used: SAM2 + DAM ({DAM_MODEL_ID})\n")
        f.write(f"Analysis Type: Grid Point Scan Only\n")
        f.write("=" * 70 + "\n\n")

        if not analysis_results:
            f.write("No significant objects or regions were identified for analysis.\n")
            return

        for i, item in enumerate(analysis_results):
            f.write(f"--- Item {i+1} ---\n")
            f.write(f"  Point of Interest: {item['point_of_interest']}\n")
            f.write(f"  Bounding Box (x1,y1,x2,y2): {item['bounding_box_xyxy']}\n")
            f.write(f"  Description (from DAM):\n")
            desc_lines = item['description'].splitlines()
            for line in desc_lines:
                f.write(f"    {line}\n")
            f.write("\n")
    print(f"Deep analysis text report (V3) saved to: {text_output_path}")


if __name__ == "__main__":
    IMAGE_FILE_PATH_TEST = r"img\1.png"

    # Create a dummy image if the specified path doesn't exist for testing purposes
    if not os.path.exists(IMAGE_FILE_PATH_TEST):
        print(f"Warning: Image path '{IMAGE_FILE_PATH_TEST}' not found. Creating a dummy image for testing.")
        dummy_img_pil = Image.new('RGB', (300, 200), color = 'lightgray')
        dummy_cv = cv2.cvtColor(np.array(dummy_img_pil), cv2.COLOR_RGB2BGR)
        cv2.rectangle(dummy_cv, (30, 30), (100, 100), (0,128,0), -1) # Green rectangle
        cv2.circle(dummy_cv, (200, 100), 40, (128,0,0), -1)      # Blue circle
        cv2.putText(dummy_cv, "Dummy V3", (50,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,50), 2)
        
        dummy_image_dir = os.path.dirname(IMAGE_FILE_PATH_TEST)
        if dummy_image_dir and not os.path.exists(dummy_image_dir):
            os.makedirs(dummy_image_dir, exist_ok=True)
        
        if not os.path.exists(os.path.dirname(IMAGE_FILE_PATH_TEST)) and os.path.dirname(IMAGE_FILE_PATH_TEST) != "":
            print(f"Original directory '{os.path.dirname(IMAGE_FILE_PATH_TEST)}' for dummy image does not exist. Saving to current directory.")
            IMAGE_FILE_PATH_TEST = "dummy_test_image_standalone_v3.png"

        cv2.imwrite(IMAGE_FILE_PATH_TEST, dummy_cv)
        print(f"Using dummy image: {IMAGE_FILE_PATH_TEST}")


    # Create output directories
    os.makedirs(OUTPUT_TEXT_FOLDER_V3, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_FOLDER_V3, exist_ok=True)
    print(f"Output text (V3) will be saved in: {OUTPUT_TEXT_FOLDER_V3}")
    print(f"Output images (V3) will be saved in: {OUTPUT_IMAGE_FOLDER_V3}")

    base_image_name = os.path.splitext(os.path.basename(IMAGE_FILE_PATH_TEST))[0]

    print(f"Testing with image: {IMAGE_FILE_PATH_TEST}")
    print("Ensure SAM2_MODEL_CFG and SAM2_CHECKPOINT paths are correct in this script.")

    if "configs/sam2.1" not in SAM2_MODEL_CFG or "checkpoints/" not in SAM2_CHECKPOINT: # Basic check
        print("WARNING: SAM2_MODEL_CFG or SAM2_CHECKPOINT might not be correctly set.")
        print(f"  Current SAM2_MODEL_CFG: {SAM2_MODEL_CFG}")
        print(f"  Current SAM2_CHECKPOINT: {SAM2_CHECKPOINT}")
        print("Please ensure they point to your actual local SAM2 model files if not using defaults.")
        # exit() # Optional: exit if paths are likely incorrect

    overall_start_time = time.time()
    analysis_results = None
    final_image_pil = None
    try:
        analysis_results, final_image_pil = analyze_image_deeply(IMAGE_FILE_PATH_TEST, auto_manage_models=True)

        if analysis_results is not None and final_image_pil is not None:
            print("\n--- Final Results Summary (Full details in text report) ---")
            for i, item in enumerate(analysis_results):
                print(f"\nItem {i+1}: Pt: {item['point_of_interest']}, Box: {item['bounding_box_xyxy']}, Desc: {item['description'][:70]}...")
            
            text_report_path_v3 = os.path.join(OUTPUT_TEXT_FOLDER_V3, f"{base_image_name}_analysis_report_standalone_v3.txt")
            save_analysis_text_v3(analysis_results, IMAGE_FILE_PATH_TEST, text_report_path_v3)

            visualization_output_path_v3 = os.path.join(OUTPUT_IMAGE_FOLDER_V3, f"{base_image_name}_visualization_standalone_v3.png")
            visualize_results(final_image_pil, analysis_results, visualization_output_path_v3)
        elif analysis_results is None and final_image_pil is None:
             print("Analysis did not complete successfully (models might not have initialized).")
        else: # Only one of them is None, or results are empty
            print("Analysis completed but yielded no descriptions or image for visualization.")
            text_report_path_v3 = os.path.join(OUTPUT_TEXT_FOLDER_V3, f"{base_image_name}_analysis_report_standalone_v3.txt")
            save_analysis_text_v3(analysis_results if analysis_results is not None else [], IMAGE_FILE_PATH_TEST, text_report_path_v3)


    except FileNotFoundError as e: print(f"ERROR: File not found: {e}")
    except ImportError as e_import: print(f"ERROR: Import failed: {e_import}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nTotal script exec time: {time.time()-overall_start_time:.2f}s")
