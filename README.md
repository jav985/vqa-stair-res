The STAIR model code is available at: https://github.com/yellow-binary-tree/STAIR/tree/main

The data folder contains outputs of script.ipynb, these can be used to run the model on the small subset we made without having to download all the data.



### 1. Download AGQA Data

* Download README and supporting data from the [AGQA homepage](https://cs.stanford.edu/people/ranjaykrishna/agqa/).
    * Unzip **`tgif-qa_frameqa_appearance_feat.h5.zip`** and **`tgif-qa_frameqa_motion_feat.h5.zip`** into `data/AGQA/video_features`
    * Unzip **scene graphs** into `data/AGQA/AGQA_scene_graphs`

### 2. Download Balanced AGQA2 and Charades Videos

* Download **balanced AGQA2 benchmark** and **CSV formatted questions**.
    * Unzip into `data/AGQA/AGQA2_balanced`
* Download **Charades videos**: [Download Charades_v1_480.zip](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip)

### 3. Download GloVe Embeddings

* Download **`glove.6B.zip`** from the [GloVe homepage](https://nlp.stanford.edu/projects/glove/).
 * Unzip **`glove.6B.zip`** to get **`glove.6B.300d.txt`** and place it in `./data/glove.6B.300d.txt`

### 4. Download I3D Features

* Download extracted **I3D features** from this [Google Drive Link for I3D features](https://drive.google.com/drive/folders/18DAXKKGckdgmIDrvJC5nYFn4tBPaWKyo?usp=sharing).
    * Unzip them to `data/AGQA/video_features/i3d_rgb` and `data/AGQA/video_features/i3d_flow`

### 5. Custom Scripts Setup

* Download **`evaluate.py`** from this repository and replace the existing one.
* Run the **`script.ipynb`** file (make sure to change paths and filenames within the notebook to match your setup).

### 6. AGQA Data Conversion

Run the following commands to convert AGQA data:
```bash
dataset=AGQA2
python -u ./utils/agqa_lite.py --func convert \
    --id2word-filename data/AGQA/AGQA_scene_graphs/ENG.txt \
    --word2id-filename data/AGQA/AGQA_scene_graphs/IDX.txt \
    --train-sg-filename data/AGQA/AGQA_scene_graphs/AGQA_train_stsgs.pkl \
    --test-sg-filename data/AGQA/AGQA_scene_graphs/AGQA_test_stsgs.pkl \
    --train-csv-filename data/AGQA/${dataset}_balanced/Train_frameqa_question-balanced.csv \
    --test-csv-filename data/AGQA/${dataset}_balanced/Test_frameqa_question-balanced.csv \
    --input-folder data/AGQA/${dataset}_balanced \
    --output-folder data/AGQA/${dataset}_balanced_pkls

```

### 7. Create Fairseq Data

Prepare data for Fairseq:

```
mkdir -p data/AGQA/AGQA2_fairseq

for split in train valid test; do \
    python utils/get_fairseq_data_from_pkl.py \
        data/AGQA/AGQA2_balanced_pkls/${split}_balanced.pkl \
        data/AGQA/AGQA2_fairseq/${split}.ques \
        ./data/AGQA/AGQA2_fairseq/${split}.prog; \
done

fairseq-preprocess \
    --trainpref data/AGQA/AGQA2_fairseq/train \
    --validpref data/AGQA/AGQA2_fairseq/valid \
    --testpref data/AGQA/AGQA2_fairseq/test \
    --source-lang ques \
    --target-lang prog \
    --destdir data/AGQA/AGQA2_fairseq/data-bin/AGQA2 \
    --dataset-impl raw \
    --workers 4

```

### 8. Train LSTM Program Parser

Train the LSTM model:

```
arch=lstm
pg_output_path=./snap/lstm-program_parser
mkdir -vp ${pg_output_path}

fairseq-train data/AGQA/AGQA2_fairseq/data-bin/AGQA2 \
    --arch lstm \
    --encoder-layers 2 \
    --decoder-layers 2 \
    --optimizer adam \
    --lr 5e-4 \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --dropout 0.2 \
    --batch-size 16 \
    --max-epoch 5 \
    --patience 10 \
    --no-progress-bar \
    --save-dir ${pg_output_path} \
    --tensorboard-logdir ${pg_output_path}/runs \
    --log-file ${pg_output_path}/log.log \
    --validate-interval-updates 900 \
    --save-interval-updates 900 \
    --keep-best-checkpoints 1 \
    --no-epoch-checkpoints \
    --tokenizer space \
    --dataset-impl raw

```

### 9. Generate Programs

Generate programs using the trained parser:

```
fairseq-generate data/AGQA/AGQA2_fairseq/data-bin/AGQA2 \
    --tokenizer space \
    --dataset-impl raw \
    --path ${pg_output_path}/checkpoint_best.pt \
    --results-path ${pg_output_path}/output \
    --batch-size 64 \
    --beam 5 \
    --nbest 5 \
    --no-progress-bar

python ./utils/agqa_lite.py --func upgrade --dataset AGQA \
    --generated-format fairseq \
    --src-data-filename data/AGQA/AGQA2_balanced_pkls/test_balanced.pkl \
    --generated-filename ${pg_output_path}/output/generate-test.txt \
    --dest-data-filename data/AGQA/AGQA2_balanced_pkls/test_balanced-generated_program.pkl

```

### 10. Training STAIR Model

Train the main STAIR model:

```
rgb_path=data/AGQA/video_features/i3d_rgb
flow_path=data/AGQA/video_features/i3d_flow
video_size=1024
max_video_length=64

stair_output_path=snap/agqa2-STAIR
mkdir -vp ${stair_output_path}

python train_module.py \
    --output ${stair_output_path} \
    --train-filename data/AGQA/AGQA2_balanced_pkls/train_balanced.pkl \
    --valid-filename data/AGQA/AGQA2_balanced_pkls/valid_balanced.pkl \
    --rgb-path $rgb_path \
    --flow-path $flow_path \
    --video-size $video_size \
    --max-video-length $max_video_length \
    --id2word-filename data/AGQA/AGQA_scene_graphs/ENG.txt \
    --word2id-filename data/AGQA/AGQA_scene_graphs/IDX.txt \
    --num-workers 0 \
    --num-epochs 5 \
    --batch-size 4 \
    --evaluate-interval 500 \
    --report-interval 100 \
    --scheduler-total-iters 80000

```

### 11. Evaluation

Evaluate the trained STAIR model:

```
python evaluate.py \
    --model-ckpt ${stair_output_path}/best_model \
    --output ${stair_output_path} \
    --result-filename result.json \
    --test-filename data/AGQA/AGQA2_balanced_pkls/test_balanced-generated_program.pkl \
    --rgb-path $rgb_path \
    --flow-path $flow_path \
    --video-size $video_size \
    --max-video-length $max_video_length \
    --id2word-filename data/AGQA/AGQA_scene_graphs/ENG.txt \
    --word2id-filename data/AGQA/AGQA_scene_graphs/IDX.txt

```

_The outputs will be saved as `results.json` in the output path you set._

### 12. Check accuracy with TensorBoard

Install TensorBoard with `pip install tensorboard` and run:

-   For program parser results:
    
    ```
    tensorboard --logdir ./snap/lstm-program_parser/runs
    
    ```
    
-   For STAIR results:
    
    ```
    tensorboard --logdir ./snap/agqa2-STAIR/runs
    
    ```
