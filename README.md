# Multi-Point Deep Image Analyzer

## Project Overview

This project provides a suite of Python scripts designed for in-depth image analysis. It leverages state-of-the-art machine learning models to perform a "sweeping analysis" of an image. By strategically selecting multiple points of interest (derived from contours and/or a grid scan), the system sequentially segments regions around these points using SAM2 (Segment Anything Model 2). An `overall_ignored_mask` is maintained to prevent re-analyzing already described areas, influencing subsequent mask generation. Textual descriptions of these segmented regions are then generated using either a local DAM (Describe Anything Model) instance or a multimodal model hosted via LMStudio.

The core idea is to go beyond single-object detection/description and provide a more holistic understanding of an image by examining multiple constituent parts, leading to significant gains in perception detail. This deep analysis can also help mitigate some limitations tied to the original training data of the description models by focusing them on specific, segmented regions.

## Features

*   **Multi-Point Sequential Analysis:** Identifies and analyzes multiple regions within a single image in a sequence, using an `overall_ignored_mask` to guide subsequent segmentations and avoid redundant processing.
*   **Advanced Segmentation:** Uses SAM2 for precise object/region segmentation based on input points.
*   **Detailed Descriptions:**
    *   Integrates with DAM (Describe Anything Model) for local, detailed descriptions of segmented masks (`deep_dam_sam2_contour.py`, `deep_dam_sam2.py`).
    *   Alternatively, integrates with LMStudio to leverage various hosted multimodal Large Language Models (LLMs) for descriptions (`deep_gguf_sam2_contour_centroid.py`).
*   **Strategic Point Selection & Ordering:**
    *   Extracts salient points from image contours (prioritizing larger objects first) and/or uses centroids (`deep_gguf_sam2_contour_centroid.py`, `deep_dam_sam2_contour.py`). This prioritizes more prominent regions before moving to potentially less critical ones.
    *   Performs a grid-based scan for comprehensive coverage, processed after contour-based points (`deep_dam_sam2.py`).
*   **Configurable Parameters:** Allows adjustment of image processing (max size), grid density, mask area thresholds, etc.
*   **Visualization:** Generates an output image annotating the points of interest, bounding boxes of described regions, and links to their descriptions. (See example below)
*   **Organized Output:** Saves detailed text reports and generated images into structured output folders.
*   **Model Management:** Includes options for automatic loading/unloading of models to manage GPU memory.
*   **Cross-Platform:** Tested and functional on both Windows and Linux.

## Example Output

*(Optional: Consider embedding a small, illustrative example image here showing the annotated output from one of your scripts. For instance, an image with a few D1, D2, D3 boxes and points, demonstrating the multi-point analysis.)*

**Example of an annotated output image:**

![example_screenshot](https://github.com/user-attachments/assets/4d1f1965-b383-4270-8179-8377f4074611)

here is one part of the output:

This image would show several highlighted regions (D1, D2, etc.) with corresponding detailed descriptions available in the generated text report. This showcases the enhanced perceptual detail achieved.

here is an extract:
--- Item 66 (Source: contour) ---
  Point of Interest: (397, 356)
  Bounding Box (x1,y1,x2,y2): [395, 304, 488, 359]
  Description:
    Here's a detailed description of the objects highlighted in the image:
    
    **Overall:** The image shows a selection of items, likely potions and equipment, from a fantasy role-playing game (likely Warcraft III). They are displayed within square frames set against a wooden background with vines. 
    
    **Individual Items (from top left to bottom right):**
    
    1.  **Scroll:** A rolled parchment tied with ribbon. It appears to be a scroll of some kind, likely containing a spell or magical effect.
    2.  **Potion/Food:** An item that looks like a piece of cooked meat, possibly a healing potion or food item for restoring health. It has a reddish-orange color and is slightly curved. 
    3.  **Amulet/Necklace:** A silver amulet with a green gem in the center. The design suggests it could be an amulet providing protection or magical benefits.
    4. **Green Potion:** A bright green potion contained within a glass bottle, likely indicating a healing or mana restoring effect. 
    5.  **Healing Potion/Flask:** A flask containing a glowing teal liquid, probably another type of healing potion or a potion with a special effect.
    6.  **Unknown Item:** The last

## Use Cases

*   **Detailed Scene Understanding:** Gaining a comprehensive understanding of complex images with multiple objects or regions of interest.
*   **Dataset Augmentation & Generation:**
    *   **Assisting Bounding Box Dataset Creation:** Quickly generate candidate bounding boxes and rich textual descriptions for objects within images, which can then be reviewed and refined to create datasets for object detection models (e.g., YOLO, Faster R-CNN).
    *   **Generating Rich Image-Text Pairs:** Create datasets for vision-language models by pairing segmented image regions with their detailed descriptions.
*   **Agentic Systems & Robotics:** Enhancing the perceptual capabilities of AI agents or robots, allowing them to identify, describe, and understand multiple elements within their environment with greater detail for better decision-making.
*   **Mitigating Training Data Biases:** By focusing description models on specific, segmented regions, this deep analysis approach can help elicit more accurate details about underrepresented objects or attributes that might be overlooked when the model processes the entire image at once due to biases in its original training data.
*   **Accessibility:** Generating rich textual descriptions of images for visually impaired users.
*   **Content Indexing and Search:** Creating detailed metadata for images, enabling more precise search and retrieval based on multiple described components.
*   **Automated Image Captioning (Granular):** Producing multiple, focused captions for different parts of an image rather than a single generic caption.
*   **Creative Content Generation:** Inspiring creative writing or art by providing detailed descriptions of image components.
*   **Educational Tools:** Helping to explain the contents of an image in a structured and detailed manner.

## Prerequisites

*   **Conda (Miniconda or Anaconda):** For environment management.
*   **Git:** Required for installing some dependencies directly from GitHub (though this has been minimized by vendoring `sam2`).
*   **NVIDIA GPU (Recommended for CUDA):** For significantly faster model inference. CPU-only execution is possible but will be very slow.
*   **NVIDIA CUDA Toolkit & cuDNN:** If using an NVIDIA GPU, ensure the appropriate CUDA Toolkit and cuDNN versions compatible with your PyTorch installation are installed.
*   **(Windows) Build Tools for Visual Studio:** May be required for compiling certain Python packages if pre-built wheels are not available. Ensure C++ build tools are installed.

## Installation

Follow these steps to set up the project environment and install dependencies.

### 1. Clone the Repository (or Unzip Your Project)

If you have this project in a Git repository:
```bash
git clone <your-repository-url>
cd <your-project-directory>
```
Otherwise, navigate to your project's root directory.

### 2. Create and Activate Conda Environment

We will create a Conda environment to isolate project dependencies. We'll name it `deep_image_analyzer`.

*   **Linux/macOS (bash/zsh):**
    ```bash
    conda create -n deep_image_analyzer python=3.10 -y
    conda activate deep_image_analyzer
    ```
*   **Windows (Anaconda Prompt or PowerShell with Conda initialized):**
    ```powershell
    conda create -n deep_image_analyzer python=3.10 -y
    conda activate deep_image_analyzer
    ```

### 3. Install PyTorch with CUDA (Recommended) or CPU

**It is highly recommended to install PyTorch *before* other dependencies, especially if you need a specific CUDA version.**

Visit the official PyTorch website to get the correct installation command for your system: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Select your preferences (OS, Package: Conda, Language: Python, Compute Platform: CUDA version or CPU).

*   **Example for CUDA 11.8 (Recommended if your GPU supports it):**
    ```bash
    # Linux/macOS or Windows (in Anaconda Prompt/Powershell)
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    ```
*   **Example for CUDA 12.1:**
    ```bash
    # Linux/macOS or Windows
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    ```
*   **Example for CPU-only (will be very slow):**
    ```bash
    # Linux/macOS or Windows
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    ```

Verify PyTorch installation and CUDA availability (if applicable) by running this Python code within your `deep_image_analyzer` Conda environment:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
```

### 4. Install Project Dependencies

Once PyTorch is installed, install the remaining dependencies using the provided `requirements.txt` file:
```bash
# Linux/macOS or Windows
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/describe-anything
```
This file includes dependencies for SAM2 (assuming it's vendored), DAM, Transformers, OpenCV, etc.

### 5. Download Model Weights and Configuration Files

The analysis scripts require pre-trained model weights and configuration files.

**SAM2 Model Files:**

You need two files for the SAM2 Hiera Large model:
*   `sam2.1_hiera_l.yaml` (Configuration file)
*   `sam2.1_hiera_large.pt` (Model weights)

**Download Links:**
*   YAML Config: [https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_l.yaml](https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_l.yaml)
*   PyTorch Weights: [https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt](https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt)

**Folder Structure:**

Create the following directory structure in your project's root and place the downloaded files accordingly:

```
<your-project-directory>/
├── configs/
│   └── sam2.1/
│       └── sam2.1_hiera_l.yaml  <-- Place YAML here
├── checkpoints/
│   └── sam2.1_hiera_large.pt    <-- Place .pt weights file here
├── sam2/                          <-- Your vendored SAM2 library code
├── dam.py                         <-- Your DAM utility script
├── deep_gguf_sam2_contour_centroid.py
├── deep_dam_sam2_contour.py
├── deep_dam_sam2.py
├── requirements.txt
└── README.md
```

You can use `wget`/`curl` (Linux/macOS) or PowerShell's `Invoke-WebRequest` (Windows), or download manually via your browser:

*   **Linux/macOS:**
    ```bash
    mkdir -p configs/sam2.1
    mkdir -p checkpoints
    wget -O configs/sam2.1/sam2.1_hiera_l.yaml https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_l.yaml
    wget -O checkpoints/sam2.1_hiera_large.pt https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt
    ```
*   **Windows (PowerShell):**
    ```powershell
    New-Item -ItemType Directory -Force -Path "configs\sam2.1"
    New-Item -ItemType Directory -Force -Path "checkpoints"
    Invoke-WebRequest -Uri https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_l.yaml -OutFile "configs\sam2.1\sam2.1_hiera_l.yaml"
    Invoke-WebRequest -Uri https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt -OutFile "checkpoints\sam2.1_hiera_large.pt"
    ```

**DAM Model:**
The DAM model (`nvidia/DAM-3B-Self-Contained`) used in `deep_dam_sam2_contour.py` and `deep_dam_sam2.py` will be downloaded automatically by the Hugging Face `transformers` library the first time it's needed. The weights will be cached locally (usually in `~/.cache/huggingface/hub/` on Linux/macOS or `C:\Users\<YourUser>\.cache\huggingface\hub\` on Windows).

**LMStudio (for `deep_gguf_sam2_contour_centroid.py`):**
If you plan to use `deep_gguf_sam2_contour_centroid.py`:
1.  Download and install LMStudio: [https://lmstudio.ai/](https://lmstudio.ai/)
2.  Within LMStudio, search for and download a multimodal model (GGUF format) capable of vision-language tasks (e.g., LLaVA, BakLLaVA, Moondream, or other compatible models).
3.  Load the downloaded model in LMStudio.
4.  Go to the "Local Server" tab (usually a `</>` icon on the left).
5.  Click "Start Server". Note the API base URL (e.g., `http://localhost:1234/v1`).
6.  **Crucially, ensure the `LMSTUDIO_API_URL` in `deep_gguf_sam2_contour_centroid.py` matches this URL.**
7.  **Also, set `LMSTUDIO_MODEL_NAME` in the script. While some LMStudio setups infer the model, others require it. You can often find the model identifier to use in the LMStudio UI when the model is loaded or by checking the API response if you try a simple curl request. If unsure, you might try leaving it as an empty string `""` first, but explicitly setting it is more reliable.**

## Running the Analysis Scripts

Ensure your `deep_image_analyzer` Conda environment is activated:
*   **Linux/macOS:** `conda activate deep_image_analyzer`
*   **Windows:** `conda activate deep_image_analyzer`

You will need to **modify the image path variable** (e.g., `IMAGE_FILE_PATH` or `IMAGE_FILE_PATH_TEST`) inside each script to point to the image you want to analyze.

1.  **`deep_gguf_sam2_contour_centroid.py` (LMStudio Integration):**
    *   Make sure LMStudio is running with a loaded multimodal model and its API server is active as described above.
    *   Verify `LMSTUDIO_API_URL` and `LMSTUDIO_MODEL_NAME` in the script.
    *   Edit the script to set `IMAGE_FILE_PATH`.
    *   Run: `python deep_gguf_sam2_contour_centroid.py`

2.  **`deep_dam_sam2_contour.py` (SAM2 + DAM with Contours & Grid):**
    *   Edit the script to set `IMAGE_FILE_PATH`.
    *   Run: `python deep_dam_sam2_contour.py`

3.  **`deep_dam_sam2.py` (SAM2 + DAM with Grid Scan - Standalone Focus):**
    *   Edit the script to set `IMAGE_FILE_PATH_TEST`.
    *   Run: `python deep_dam_sam2.py`

### Output

The scripts will generate:
*   Console output with progress and summaries.
*   A main output folder (e.g., `analysis_output`, `analysis_output_sam2_dam`, `analysis_output_standalone_v3`).
*   Inside this, a `texts/` subfolder containing detailed `.txt` reports of the analysis.
*   Inside this, an `images/` subfolder containing:
    *   The visualization image (e.g., `your_image_visualization.png`).
    *   A debug image showing contours and centroids (for scripts using contour analysis).

## Configuration

Key configuration parameters are located at the top of each script. These include:
*   `MAX_IMAGE_WIDTH`, `MAX_IMAGE_HEIGHT`: To resize large input images.
*   `NUM_POINTS_HORIZONTAL`, `NUM_POINTS_VERTICAL`: For grid scan density.
*   `MIN_MASK_AREA_THRESHOLD`: To filter out very small, insignificant masks.
*   `MAX_TOTAL_CONTOUR_POINTS`: Limits points from contour analysis.
*   `SAM2_MODEL_CFG`, `SAM2_CHECKPOINT`: Paths to SAM2 model files.
*   `DAM_MODEL_ID`, `DAM_QUERY`: For DAM model.
*   `LMSTUDIO_API_URL`, `LMSTUDIO_MODEL_NAME`: For LMStudio integration. **Ensure these are correctly set for `deep_gguf_sam2_contour_centroid.py`.**
*   Output folder names.

Adjust these as needed for your specific requirements and image characteristics.

## Limitations and Potential Improvements

### Current Limitations

*   **Description Quality & Training Data Bottlenecks:**
    *   The accuracy and detail of the generated descriptions are highly dependent on the capabilities and original training data of the chosen description model (DAM, or the specific model used via LMStudio). Biases or gaps in the model's training can affect description quality for certain objects or scenes.
    *   However, this project's approach of deep analysis on segmented regions can *help mitigate* some of these issues by forcing the model to focus on smaller, specific parts, potentially revealing details it might otherwise miss in a global image view.
*   **Computational Cost & Speed:**
    *   Analyzing images with many points can be time-consuming, as SAM2 segmentation for each point and subsequent description generation are processed sequentially.
    *   GPU VRAM can be a limiting factor, especially with large models like DAM-3B.
*   **Sequential SAM2 Processing:** SAM2 segmentation is performed sequentially for each point. This is inherent to the current design, which uses an `overall_ignored_mask` to prevent re-segmenting already analyzed areas. The mask generated for one point influences the available area for subsequent points.
*   **Mask Quality from SAM2:** While SAM2 is powerful, the quality of segmentation can vary. Imperfect masks can lead to less accurate descriptions. The sequential nature means an early, overly large mask could inadvertently obscure parts of other objects intended for later analysis.
*   **Handling of Overlapping Regions:**
    *   The system analyzes distinct regions based on different input points. If these regions (and their bounding boxes) overlap, each is described independently based on its specific mask.
    *   For the **DAM scripts**, the full image along with a specific binary mask for the current region is sent to the DAM model.
    *   For the **LMStudio script**, the original image is masked (non-selected areas become transparent), and then this masked image is *cropped to the bounding box of the current segment* before being sent to the LMStudio model.
    *   In both cases, the models are guided to describe the *masked content*. The system does not currently perform explicit reasoning about the relationships *between* overlapping described segments.
*   **Point Selection Strategy & Order Dependency:**
    *   The current strategy (contours prioritized by area, then grid) aims to analyze more prominent regions first. This is important because the `overall_ignored_mask` means the order of point processing can influence the final set of masks and descriptions, as earlier masks "claim" pixels.
    *   While the goal is to capture all relevant regions eventually, the strategy might not be universally optimal for all image types or desired levels of detail for every part of an image.

### Potential Future Improvements

*   **Parallel Inference for Descriptions (Post-SAM2):**
    *   While SAM2 processing is sequential due to the `overall_ignored_mask`, once all desired masks are generated, the *description generation step* for these multiple, independent (mask, image patch) pairs could be parallelized if sufficient GPU memory and compute resources are available. This would involve:
        *   Collecting all (image patch/full image + mask) data after the SAM2 loop.
        *   Batching these for parallel calls to the description model (DAM or LMStudio).
*   **Advanced Point/Region Proposal & Prioritization:**
    *   Integrate more sophisticated object proposal networks as an alternative or supplement to current point generation strategies.
    *   Allow users to interactively select points or draw rough bounding boxes.
    *   Explore more dynamic prioritization schemes for point processing based on initial low-cost image analysis.
*   **Hierarchical Analysis & Relationship Modeling:**
    *   After describing individual regions, an additional LLM step could be used to synthesize a summary that describes the relationships between the identified objects/regions.
*   **Description Refinement & Consistency:**
    *   Employ techniques to ensure consistency in terminology across descriptions for the same image.
*   **Optimized Model Usage:**
    *   Explore model quantization or distillation for DAM/LMStudio models to reduce resource requirements.
*   **User Interface:** Develop a Gradio or web-based UI for easier image uploading, parameter tuning, interactive point selection, and result exploration.
*   **Feedback Loop for Segmentation/Description:** If a description seems nonsensical or a mask is poor, allow a mechanism to try re-segmenting with different parameters or points for that region, or re-prompting the description model.
*   **Adaptive `overall_ignored_mask` Behavior:** Explore options for more nuanced updates to the `overall_ignored_mask`, perhaps allowing some degree of overlap or refinement if a later point strongly suggests a different segmentation for an already partially masked area. This is complex but could improve results in dense scenes.
*   **Support for Video Analysis:** Extend the "sweeping analysis" concept to video frames, leveraging SAM2's video capabilities for tracking and describing changes over time.

## Troubleshooting

*   **`FileNotFoundError` for SAM2 models:** Double-check the paths in the scripts (`SAM2_MODEL_CFG`, `SAM2_CHECKPOINT`) and ensure the `.yaml` and `.pt` files are correctly placed in the `configs/sam2.1/` and `checkpoints/` directories respectively.
*   **CUDA Errors / Out of Memory:**
    *   Ensure your PyTorch and CUDA versions are compatible.
    *   If you have multiple GPUs, you might need to set `CUDA_VISIBLE_DEVICES` environment variable.
    *   Reduce `MAX_IMAGE_WIDTH` / `MAX_IMAGE_HEIGHT` if processing very large images.
    *   The models (especially DAM-3B) are large. Ensure you have sufficient GPU VRAM (and system RAM). Script `deep_dam_sam2.py` includes model unloading capabilities which can be helpful.
*   **Slow Performance:** CPU execution will be very slow. Using a CUDA-enabled GPU is highly recommended.
*   **Import Errors for `sam2` or `dam`:**
    *   Ensure the `sam2` folder (your vendored copy) and `dam.py` are in the project's root directory or accessible via your Python path.
    *   Make sure you are in the correct `deep_image_analyzer` Conda environment.
*   **LMStudio Connection Issues (`deep_gguf_sam2_contour_centroid.py`):**
    *   Verify LMStudio is running and the local server is active on the correct port.
    *   Check that the `LMSTUDIO_API_URL` in the script matches the one shown in LMStudio.
    *   Ensure a compatible multimodal model is loaded and selected for the API in LMStudio. The model selected for the API might be different from the one just loaded in the chat UI.
    *   Test the LMStudio API endpoint with a tool like `curl` or Postman to ensure it's responding correctly before running the script. Example `curl` (replace with your model if needed):
        ```bash
        curl http://localhost:1234/v1/chat/completions -H "Content-Type: application/json" -d '{ "model": "your-loaded-model-id-if-needed", "messages": [{"role": "user", "content": "What is in this image?"}], "max_tokens": 50 }'
        ```
*   **(Windows) Compilation errors during `pip install`:** Ensure you have "Build Tools for Visual Studio" with the "C++ build tools" workload installed. Some packages might need to compile C/C++ extensions.

## License

This project's scripts are provided as-is. The licenses of the underlying models (SAM2, DAM, and any models used via LMStudio) apply to their respective components. Please consult their original sources for licensing information.
*   SAM2: Typically Apache 2.0 or similar (check Facebook Research).
*   DAM (nvidia/DAM-3B-Self-Contained): Check Hugging Face model card for license.
*   Models from LMStudio: Vary by model.
