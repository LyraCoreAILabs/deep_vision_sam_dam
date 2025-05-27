import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
import time
import os
import tempfile
import requests # For LMStudio API calls
import base64
import io
import json

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError as e:
    print(f"Error importing SAM2 utilities: {e}")
    print("Please ensure 'sam2' library is installed.")
    print("For SAM2, you might need to clone its repository and set PYTHONPATH.")
    exit()

# --- Configuration ---
MAX_IMAGE_WIDTH = 640
MAX_IMAGE_HEIGHT = 480
NUM_POINTS_HORIZONTAL = 14
NUM_POINTS_VERTICAL = 12
MIN_MASK_AREA_THRESHOLD = 50
MAX_TOTAL_CONTOUR_POINTS = 500 # Upper limit for points from contours

# SAM2 Configuration
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml" # UPDATE IF NEEDED
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt" # UPDATE IF NEEDED

# LMStudio Configuration
LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions" # Default LMStudio endpoint
LMSTUDIO_MODEL_NAME = "mlabonne_gemma-3-27b-it-abliterated" # Example, replace with your loaded model's identifier or ""
LMSTUDIO_PROMPT = "Describe the primary object or region highlighted in this image in detail."

# Output Folder Configuration
OUTPUT_BASE_FOLDER = "analysis_output"
OUTPUT_TEXT_FOLDER = os.path.join(OUTPUT_BASE_FOLDER, "texts")
OUTPUT_IMAGE_FOLDER = os.path.join(OUTPUT_BASE_FOLDER, "images")

# Global model instances and device
sam2_predictor = None
device = None
sam2_initialized = False # Flag to track SAM2 initialization

def initialize_sam2(cfg_path=SAM2_MODEL_CFG, ckpt_path=SAM2_CHECKPOINT):
    global sam2_predictor, device, sam2_initialized
    if sam2_initialized:
        print("SAM2 already initialized.")
        return

    print("Initializing SAM2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading SAM2 model with config: {cfg_path} and checkpoint: {ckpt_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"SAM2 model config not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")
    sam2_predictor = build_sam2_video_predictor(cfg_path, ckpt_path, device=device)
    print("SAM2 predictor loaded.")
    sam2_initialized = True
    print("SAM2 initialized successfully.")

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

def get_centroids_and_contours_from_image(image_pil, max_total_centroid_points):
    if image_pil.width <= 0 or image_pil.height <= 0:
        print("Warning: Cannot get contours from an image with non-positive dimensions.")
        return [], []

    img_cv_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    height, width = img_cv_bgr.shape[:2]
    img_cv_gray = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_cv_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    all_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potential_candidates = []
    for contour in all_contours:
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cX = max(0, min(cX, width - 1))
            cY = max(0, min(cY, height - 1))
            potential_candidates.append({'contour': contour, 'area': area, 'centroid': (cX, cY)})

    potential_candidates.sort(key=lambda c: c['area'], reverse=True)
    selected_centroid_points = []
    selected_contours_for_debug = []

    for candidate in potential_candidates:
        if len(selected_centroid_points) < max_total_centroid_points:
            if candidate['centroid'] not in selected_centroid_points:
                selected_centroid_points.append(candidate['centroid'])
                selected_contours_for_debug.append(candidate['contour'])
        else:
            break
    print(f"Found {len(all_contours)} initial contours. Identified {len(potential_candidates)} valid candidates.")
    print(f"Selected {len(selected_centroid_points)} unique centroids (limit: {max_total_centroid_points}).")
    return selected_centroid_points, selected_contours_for_debug

def apply_sam2_for_point(inference_state, point_xy, input_label=1):
    global sam2_predictor, device
    if not sam2_initialized or sam2_predictor is None:
        raise ValueError("SAM2 predictor not initialized. Call initialize_sam2() first.")
    if inference_state is None:
        raise ValueError("SAM2 inference_state not initialized.")

    points_np = np.array([[int(point_xy[0]), int(point_xy[1])]], dtype=np.float32)
    labels_np = np.array([input_label], dtype=np.int32)

    sam2_predictor.reset_state(inference_state) # Reset for each new point query
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
                    single_mask_logits = single_mask_logits.squeeze(0) if single_mask_logits.shape[0] == 1 else single_mask_logits[0]
                mask_tensor = (single_mask_logits > 0.0).cpu()
                if mask_tensor.ndim != 2:
                    print(f"Warning: Mask tensor from SAM2 is not 2D after processing! Shape: {mask_tensor.shape}")
                    return np.array([], dtype=bool)
                mask_np = mask_tensor.numpy().astype(bool)
            else:
                print("SAM2 did not return mask logits for the point.")
    return mask_np

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

def get_description_from_lmstudio(original_image_pil, mask_np, prompt_text, bbox_xyxy):
    if not np.any(mask_np):
        return "No significant region found by SAM2 to describe."

    if mask_np.shape != (original_image_pil.height, original_image_pil.width):
        print(f"Warning: Mask shape {mask_np.shape} mismatch with image H,W {(original_image_pil.height, original_image_pil.width)}. Resizing mask to image dimensions.")
        mask_pil_for_resize = Image.fromarray(mask_np.astype(np.uint8) * 255)
        mask_pil_resized = mask_pil_for_resize.resize(original_image_pil.size, Image.Resampling.NEAREST)
        mask_np_final = np.array(mask_pil_resized) > 0 
    else:
        mask_np_final = mask_np

    img_rgba = original_image_pil.convert("RGBA")
    mask_pil_alpha = Image.fromarray(mask_np_final.astype(np.uint8) * 255, 'L')
    img_rgba.putalpha(mask_pil_alpha)

    if bbox_xyxy:
        x1, y1, x2, y2 = bbox_xyxy
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(original_image_pil.width, x2)
        y2 = min(original_image_pil.height, y2)
        if x1 >= x2 or y1 >= y2:
             print(f"Warning: Invalid bounding box after clipping: {[x1,y1,x2,y2]}. Using full masked image.")
             cropped_image = img_rgba 
        else:
            cropped_image = img_rgba.crop((x1,y1,x2,y2))
    else: 
        print("Warning: Bounding box not provided to get_description_from_lmstudio. Using full masked image.")
        cropped_image = img_rgba
    
    if cropped_image.width == 0 or cropped_image.height == 0:
        return "Cropped image region is empty after applying mask and bbox."

    buffered = io.BytesIO()
    cropped_image.save(buffered, format="PNG") 
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ],
        "max_tokens": 256, 
        "temperature": 0.3, 
        "stream": False
    }
    if LMSTUDIO_MODEL_NAME and LMSTUDIO_MODEL_NAME.strip() != "":
        payload["model"] = LMSTUDIO_MODEL_NAME

    print(f"Sending request to LMStudio: {LMSTUDIO_API_URL} for a {cropped_image.width}x{cropped_image.height} cropped region.")
    try:
        response = requests.post(LMSTUDIO_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0:
            content = response_json["choices"][0].get("message", {}).get("content")
            if content:
                return content.strip()
            else:
                return "Error: No content in LMStudio response choice."
        else:
            return f"Error: Unexpected response structure from LMStudio: {response_json}"
    except requests.exceptions.RequestException as e:
        return f"Error communicating with LMStudio: {e}"
    except json.JSONDecodeError:
        return f"Error: Could not decode JSON response from LMStudio. Response text: {response.text}"
    except Exception as e:
        return f"An unexpected error occurred with LMStudio request: {e}"


def visualize_results(image_pil, results_data, output_path):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.4
    font_color_point_contour = (255,0,255); 
    font_color_point_grid = (0,0,255);     
    font_color_box = (0,255,0); thickness = 1 
    drawn_text_bboxes = []

    for i, item in enumerate(results_data):
        point_coords = item['point_of_interest']; box = item['bounding_box_xyxy']
        source = item.get('source', 'grid') 
        px, py = point_coords
        point_color = font_color_point_contour if source == 'contour' else font_color_point_grid
        cv2.circle(img_cv, (px,py), 3, point_color, -1)

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
    print("\nVisualization Key:")
    for i, item in enumerate(results_data):
        source = item.get('source', 'grid')
        print(f"P{i+1} (Point from {source}): {item['point_of_interest']}")
        print(f"D{i+1} (Description): {item['description'][:70]}...")

def analyze_image_deeply(image_path):
    global sam2_predictor, sam2_initialized 

    if not sam2_initialized:
        initialize_sam2() 
    
    if not sam2_initialized or sam2_predictor is None: 
        print("Error: SAM2 predictor failed to initialize. Aborting analysis.")
        return [], None, [], []

    processed_image_pil = preprocess_image(image_path, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)
    img_width, img_height = processed_image_pil.size

    if img_width <= 0 or img_height <= 0:
        print(f"ERROR: Processed image has invalid dimensions: {img_width}x{img_height}. Aborting analysis.")
        return [], processed_image_pil, [], []

    overall_ignored_mask = np.zeros((img_height, img_width), dtype=bool)
    results = []
    extracted_centroid_points = []
    all_selected_contours_for_debug = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_image_filename = "0000.jpg"
        temp_image_path = os.path.join(temp_dir, temp_image_filename)
        processed_image_pil.save(temp_image_path, "JPEG") 
        print(f"Saved temporary image for SAM2: {temp_image_path}")

        print("Initializing SAM2 inference state (this can be slow for the first image)...")
        start_init_time = time.time()
        sam_inference_state = sam2_predictor.init_state(video_path=temp_dir)
        print(f"SAM2 inference state initialized in {time.time()-start_init_time:.2f}s.")

        points_to_process = []
        point_sources = []

        print("\n--- Stage 1: Processing points from contours ---")
        extracted_centroid_points, all_selected_contours_for_debug = get_centroids_and_contours_from_image(
            processed_image_pil, MAX_TOTAL_CONTOUR_POINTS
        )
        if extracted_centroid_points:
            points_to_process.extend(extracted_centroid_points)
            point_sources.extend(['contour'] * len(extracted_centroid_points))

        print("\n--- Stage 2: Processing points from grid scan (sweeping analysis) ---")
        grid_points = generate_grid_points(img_width,img_height,NUM_POINTS_HORIZONTAL,NUM_POINTS_VERTICAL)
        points_to_process.extend(grid_points)
        point_sources.extend(['grid'] * len(grid_points))

        unique_points_map = {}
        final_points_to_process = []
        final_point_sources = []
        for i, pt_tuple in enumerate(points_to_process):
            if pt_tuple not in unique_points_map:
                unique_points_map[pt_tuple] = point_sources[i]
                final_points_to_process.append(pt_tuple)
                final_point_sources.append(point_sources[i])

        points_to_process = final_points_to_process
        point_sources = final_point_sources
        total_points_to_process = len(points_to_process)
        print(f"Total unique points to process (contours prioritized, then grid): {total_points_to_process}")

        analysis_loop_start_time = time.time()

        for i, point_xy_tuple in enumerate(points_to_process):
            px, py = int(point_xy_tuple[0]), int(point_xy_tuple[1])
            source_type = point_sources[i]
            print(f"\nPoint {i+1}/{total_points_to_process} from {source_type}: ({px},{py})")

            if not (0 <= px < img_width and 0 <= py < img_height):
                print(f"ERROR: Point ({px},{py}) is out of image bounds ({img_width}x{img_height}). Skipping.")
                continue
            if overall_ignored_mask[py, px]:
                print("Point is within an already described (ignored) region. Skipping.")
                continue

            point_proc_start_time = time.time()
            print("Applying SAM2...")
            current_mask_np = apply_sam2_for_point(sam_inference_state, point_xy_tuple)

            if current_mask_np.ndim == 0 or current_mask_np.size == 0:
                print("SAM2 returned an empty array (no mask). Skipping.")
                continue
            if current_mask_np.ndim != 2:
                print(f"ERROR: Mask from SAM2 is not 2D! Shape: {current_mask_np.shape}. Skipping.")
                continue
            
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
                print(f"SAM2 did not produce a significant mask (Area: {current_mask_np.sum()}, Threshold: {MIN_MASK_AREA_THRESHOLD}).")
                continue

            bounding_box = get_bounding_box_from_mask(current_mask_np)
            if bounding_box:
                print(f"SAM2 produced mask. Bounding box: {bounding_box}")
            else:
                print(f"SAM2 mask generated, but no valid bounding box. Area: {current_mask_np.sum()}, Shape: {current_mask_np.shape}")
                continue 

            print("Getting description from LMStudio...")
            description = get_description_from_lmstudio(processed_image_pil, current_mask_np, LMSTUDIO_PROMPT, bounding_box)
            print(f"Description: {description[:100]}...")
            
            results.append({
                "point": point_xy_tuple,
                "description": description,
                "box": bounding_box,
                "mask_sum": int(current_mask_np.sum()),
                "source": source_type
            })
            overall_ignored_mask = np.logical_or(overall_ignored_mask, current_mask_np)
            print(f"Point processing time: {time.time()-point_proc_start_time:.2f}s. Overall ignored: {overall_ignored_mask.sum()/overall_ignored_mask.size*100:.2f}%")

        print(f"\n--- Analysis Loop Complete ---")
        print(f"Total point processing loop time: {time.time()-analysis_loop_start_time:.2f}s. Found {len(results)} descriptions.")

    final_output_list_for_visualization = []
    for res_item in results:
        final_output_list_for_visualization.append({
            "description": res_item["description"],
            "point_of_interest": res_item["point"],
            "bounding_box_xyxy": res_item["box"],
            "source": res_item["source"]
        })
    return final_output_list_for_visualization, processed_image_pil, extracted_centroid_points, all_selected_contours_for_debug

def save_contour_centroid_debug_image(image_pil, selected_contours, centroid_points, output_path):
    if not image_pil:
        print("No image provided for debug visualization.")
        return
    if not selected_contours and not centroid_points:
        print("No contours or centroids to draw for debug image.")
        return

    img_cv_bgr = cv2.cvtColor(np.array(image_pil.copy()), cv2.COLOR_RGB2BGR)
    if selected_contours:
        print(f"Drawing {len(selected_contours)} selected contours for debug image.")
        for i, contour in enumerate(selected_contours):
            color = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)][i % 6]
            cv2.drawContours(img_cv_bgr, [contour], -1, color, 1)
    if centroid_points:
        print(f"Drawing {len(centroid_points)} selected centroids for debug image.")
        for i, (cX, cY) in enumerate(centroid_points):
            cv2.circle(img_cv_bgr, (cX, cY), 3, (0, 255, 0), -1) 
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        cv2.imwrite(output_path, img_cv_bgr)
        print(f"Saved contour/centroid debug visualization to: {output_path}")
    except Exception as e_save:
        print(f"Error saving contour/centroid debug visualization: {e_save}")

def save_analysis_text(analysis_results, input_image_filename, text_output_path):
    """Saves the detailed analysis results to a text file."""
    os.makedirs(os.path.dirname(text_output_path), exist_ok=True) # Ensure directory exists
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write(f"Deep Analysis Report for Image: {os.path.basename(input_image_filename)}\n")
        f.write("=" * 70 + "\n\n")

        if not analysis_results:
            f.write("No significant objects or regions were identified for analysis.\n")
            return

        for i, item in enumerate(analysis_results):
            source_info = f" (Source: {item.get('source', 'N/A')})"
            f.write(f"--- Item {i+1}{source_info} ---\n")
            f.write(f"  Point of Interest: {item['point_of_interest']}\n")
            f.write(f"  Bounding Box (x1,y1,x2,y2): {item['bounding_box_xyxy']}\n")
            f.write(f"  Description:\n")
            # Indent description for better readability
            desc_lines = item['description'].splitlines()
            for line in desc_lines:
                f.write(f"    {line}\n")
            f.write("\n") # Extra newline for separation
    print(f"Deep analysis text report saved to: {text_output_path}")


if __name__ == "__main__":
    IMAGE_FILE_PATH = r"img\1.png" 
    # Example: IMAGE_FILE_PATH = "path/to/your/image.jpg"

    if not os.path.exists(IMAGE_FILE_PATH):
        print(f"Warning: Image path '{IMAGE_FILE_PATH}' not found. Creating a dummy image for testing.")
        dummy_img_pil = Image.new('RGB', (300, 200), color = 'lightgray')
        dummy_cv = cv2.cvtColor(np.array(dummy_img_pil), cv2.COLOR_RGB2BGR)
        cv2.rectangle(dummy_cv, (30, 30), (100, 100), (0,128,0), -1) 
        cv2.circle(dummy_cv, (200, 100), 40, (128,0,0), -1)      
        cv2.line(dummy_cv, (10, 150), (290, 160), (0,0,128), 3) 
        cv2.putText(dummy_cv, "Dummy", (50,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,50), 2)
        
        # Ensure the 'img' directory exists for the dummy image or save it in current dir
        dummy_image_dir = os.path.dirname(IMAGE_FILE_PATH)
        if dummy_image_dir and not os.path.exists(dummy_image_dir):
            os.makedirs(dummy_image_dir, exist_ok=True)
        
        # If original IMAGE_FILE_PATH was in a non-existent dir, save dummy to current dir
        if not os.path.exists(os.path.dirname(IMAGE_FILE_PATH)) and os.path.dirname(IMAGE_FILE_PATH) != "":
            print(f"Original directory '{os.path.dirname(IMAGE_FILE_PATH)}' for dummy image does not exist. Saving to current directory.")
            IMAGE_FILE_PATH = "dummy_test_image_combined.png"

        cv2.imwrite(IMAGE_FILE_PATH, dummy_cv)
        print(f"Using dummy image: {IMAGE_FILE_PATH}")

    # Create output directories if they don't exist
    os.makedirs(OUTPUT_TEXT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
    print(f"Output text will be saved in: {OUTPUT_TEXT_FOLDER}")
    print(f"Output images will be saved in: {OUTPUT_IMAGE_FOLDER}")

    base_image_name = os.path.splitext(os.path.basename(IMAGE_FILE_PATH))[0]

    print(f"Starting analysis for image: {IMAGE_FILE_PATH}")
    print("Ensure SAM2_MODEL_CFG and SAM2_CHECKPOINT paths are correct.")
    print(f"Ensure LMStudio is running with a multimodal model, API enabled at '{LMSTUDIO_API_URL}'.")
    if LMSTUDIO_MODEL_NAME:
        print(f"Attempting to use LMStudio model: '{LMSTUDIO_MODEL_NAME}' (if specified).")
    
    if "configs/sam2.1" not in SAM2_MODEL_CFG or "checkpoints/" not in SAM2_CHECKPOINT:
         print("WARNING: SAM2_MODEL_CFG or SAM2_CHECKPOINT might not be correctly set in the script.")
         print(f"  Current SAM2_MODEL_CFG: {SAM2_MODEL_CFG}")
         print(f"  Current SAM2_CHECKPOINT: {SAM2_CHECKPOINT}")
         print("Please ensure they point to your actual local SAM2 model files if not using defaults from a cloned SAM2 repo.")


    overall_start_time = time.time()
    analysis_results = []
    processed_pil_image_for_debug = None
    centroid_points_for_debug = []
    selected_contours_for_debug = []

    try:
        analysis_results, processed_pil_image_for_debug, centroid_points_for_debug, selected_contours_for_debug = \
            analyze_image_deeply(IMAGE_FILE_PATH)

        if analysis_results:
            print("\n--- Final Results List ---")
            for i, item in enumerate(analysis_results):
                source_info = f" (Source: {item.get('source', 'N/A')})"
                print(f"\nItem {i+1}{source_info}:")
                print(f"  Point of Interest: {item['point_of_interest']}")
                print(f"  Bounding Box (x1,y1,x2,y2): {item['bounding_box_xyxy']}")
                print(f"  Description: {item['description']}")
            
            # Save the full text analysis
            text_report_path = os.path.join(OUTPUT_TEXT_FOLDER, f"{base_image_name}_analysis_report.txt")
            save_analysis_text(analysis_results, IMAGE_FILE_PATH, text_report_path)

            if processed_pil_image_for_debug:
                visualization_output_path = os.path.join(OUTPUT_IMAGE_FOLDER, f"{base_image_name}_visualization.png")
                visualize_results(processed_pil_image_for_debug, analysis_results, visualization_output_path)
        else:
            print("\nNo descriptions were generated from the image analysis.")
            # Save an empty report if no results
            text_report_path = os.path.join(OUTPUT_TEXT_FOLDER, f"{base_image_name}_analysis_report.txt")
            save_analysis_text([], IMAGE_FILE_PATH, text_report_path)


        if processed_pil_image_for_debug and (centroid_points_for_debug or selected_contours_for_debug):
            contour_debug_output_path = os.path.join(OUTPUT_IMAGE_FOLDER, f"{base_image_name}_contours_debug.png")
            save_contour_centroid_debug_image(
                processed_pil_image_for_debug,
                selected_contours_for_debug,
                centroid_points_for_debug,
                contour_debug_output_path
            )
        elif processed_pil_image_for_debug:
             print("Contour/centroid debug image not saved as no contours/centroids were selected/found.")
        else:
            print("Could not generate contour debug image as processed image was not available.")

    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found: {e}")
        print("Please check image paths, and SAM2 model/config paths.")
    except ImportError as e_import:
        print(f"ERROR: Failed to import a required library: {e_import}")
        print("Ensure SAM2, PyTorch, OpenCV, etc., and their dependencies are correctly installed.")
    except requests.exceptions.ConnectionError as e_conn:
        print(f"ERROR: Could not connect to LMStudio API at {LMSTUDIO_API_URL}.")
        print("Please ensure LMStudio is running, the server is enabled, and the URL is correct.")
        print(f"Details: {e_conn}")
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nTotal script execution time: {time.time()-overall_start_time:.2f} seconds")