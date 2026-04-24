# KVBench

KVBench is a comprehensive benchmark for evaluating knowledge visualization capabilities of text-to-image models across multiple scientific disciplines including Physics, Chemistry, Biology, Geography, Mathematics, and History.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Supported Subjects](#supported-subjects)
- [Notes](#notes)

## Project Structure
```
KVBench/
|-- data/
|   |-- Physics/                    # Physics dataset images
|   |-- Physics_English/            # Physics dataset images (English)
|   |-- Chemistry/                  # Chemistry dataset images  
|   |-- Chemistry_English/          # Chemistry dataset images (English)
|   |-- Biology/                    # Biology dataset images
|   |-- Biology_English/            # Biology dataset images (English)
|   |-- Geography/                  # Geography dataset images
|   |-- Geography_English/          # Geography dataset images (English)
|   |-- Mathematics/                # Mathematics dataset images
|   |-- Mathematics_English/        # Mathematics dataset images (English)
|   |-- History/                    # History dataset images
|   |-- History_English/            # History dataset images (English)
|   |-- annotations/                # Original .jsonl annotation files
|   `-- results/                    # Experimental Outputs
|       `-- {model_name}_{source}/  # e.g., flux_brief
|           |-- Physics/            # Generated physics images
|           |-- Physics_English/    # Generated physics images (English)
|           |-- Chemistry/          # Generated chemistry images
|           |-- Chemistry_English/  # Generated chemistry images (English)
|           |-- Biology/            # Generated biology images
|           |-- Biology_English/    # Generated biology images (English)
|           |-- Geography/          # Generated geography images
|           |-- Geography_English/  # Generated geography images (English)
|           |-- Mathematics/        # Generated mathematics images
|           |-- Mathematics_English/# Generated mathematics images (English)
|           |-- History/            # Generated history images
|           `-- History_English/    # Generated history images (English)
|-- scripts/                        
|   `-- Evaluation.py               # Automated evaluation pipeline
|-- models/                         # Local Weights (Excluded from Git)
|   `-- Qwen/Qwen2.5-VL-32B-Instruct/    # Evaluation model weights
|-- requirements.txt                # Python dependencies
`-- README.md                       # Project Documentation
```

## Installation

### Step 1: Navigate to project directory
```bash
cd KVBench
```

### Step 2: Create conda environment
```bash
conda create -n kvbench python=3.10
conda activate kvbench
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Evaluation Pipeline

Run automated evaluation on generated images using Qwen2.5-VL model:

```bash
python scirpts/Evaluation.py \
  --subject physics \
  --path_name path \
  --model_path ./models/Qwen/Qwen2.5-VL-32B-Instruct
```

Evaluate multiple subjects:
```bash
python scripts/Evaluation.py \
  --subject physics,chemistry,biology \
  --path_name path \
  --model_path ./models/Qwen/Qwen2.5-VL-32B-Instruct
```

## Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--subject` | Single or multiple subjects (comma-separated) | `--subject physics` or `--subject physics,chemistry` |
| `--path_name` | Path to the dataset directory | `--path_name ./data` |
| `--model_path` | Path to evaluation model weights | `--model_path ./models/Qwen/Qwen2.5-VL-32B-Instruct` |

## Supported Subjects

### Chinese
- `physics` - Physics concepts and phenomena
- `chemistry` - Chemical structures and reactions
- `biology` - Biological processes and structures  
- `geography` - Geographical features and phenomena 
- `mathematics` - Mathematical diagrams and concepts
- `history` - Historical events and figures

### English
- `physics_English` - Physics concepts and phenomena
- `chemistry_English` - Chemical structures and reactions 
- `biology_English` - Biological processes and structures  
- `geography_English` - Geographical features and phenomena
- `mathematics_English` - Mathematical diagrams and concepts
- `history_English` - Historical events and figures

## Notes

- Download Qwen2.5-VL-32B-Instruct model from https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct and place it in `./models/Qwen/Qwen2.5-VL-32B-Instruct` directory

## License
KVBench is licensed under Apache 2.0.
