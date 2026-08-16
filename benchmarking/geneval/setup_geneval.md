
## Environment Setup for GenEval

Follow the commands to set up the conda environment

```bash
conda create -n evalenv6 python=3.8

./evaluation/download_models.sh "<OBJECT_DETECTOR_FOLDER>/"

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install open-clip-torch==2.26.1
pip install clip-benchmark
pip install -U openmim
pip install einops
python -m pip install lightning
pip install 'diffusers[torch]' transformers
pip install tomli
pip install platformdirs
pip install --upgrade setuptools

mim install mmengine mmcv-full==1.7.2

git clone -b 2.x https://github.com/open-mmlab/mmdetection.git  # 2.x branch: evaluation code requires mmdet v2 API (matches mmcv-full 1.7.2)
cd mmdetection
pip install -v -e .
```

## Running GenEval

After generating images with the model of your choice, run the evaluation using:

```bash
./run_geneval.sh <INPUT_DIR> <OUTPUT_JSONL>
```

* `<INPUT_DIR>`: Directory containing generated images in GenEval format.
* `<OUTPUT_JSONL>`: Name of the file where evaluation results will be saved.


NOTE: This works with A100 node, not with H100. 

NOTE 2: Utilize the metadata files under `prompts` folder to generate images.