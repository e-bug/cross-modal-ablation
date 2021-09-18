# Cross-Modal Ablation

This is the implementation of the approaches described in the paper:

> Stella Frank*, Emanuele Bugliarello* and Desmond Elliott. [Vision-_and_-Language or Vision-_for_-Language? On Cross-Modal Influence in Multimodal Transformers](https://arxiv.org/abs/2109.04448). In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP), Nov 2021.

We provide the code for reproducing our results.

The `cross-modal ablation task` has also been integrated into [VOLTA](https://github.com/e-bug/volta), upon which our repository was origally built.


## Repository Setup

You can clone this repository issuing:<br>
`git clone git@github.com:e-bug/cross-modal-ablation`

1\. Create a fresh conda environment, and install all dependencies.
```text
conda create -n xm-influence python=3.6
conda activate xm-influence
pip install -r requirements.txt
```

2\. Install [apex](https://github.com/NVIDIA/apex).
If you use a cluster, you may want to first run commands like the following:
```text
module load cuda/10.1.105
module load gcc/8.3.0-cuda
```

3\. Setup the `refer` submodule for Referring Expression Comprehension:
```
cd tools/refer; make
```

## Data

For textual data, please clone the Flickr30K Entities repository: <br>
`git@github.com:BryanPlummer/flickr30k_entities.git`

For visual features, we use the [VOLTA release for Flickr30K](https://sid.erda.dk/sharelink/CrLpUMgIKh).

Our datasets directory looks as follows:
```text
data/
 |-- flickr30k/
 |    |-- resnet101_faster_rcnn_genome_imgfeats/
 |
 |-- flickr30k_entities/
 |    |-- Annotations/
 |    |-- Sentences/
 |    |-- val.txt
```

Once you have defined the path to your datasets directory, make sure to update the cross-modal influence configuration file (e.g. [`volta/config_tasks/xm-influence_test_tasks.yaml`](volta/config_tasks/xm-influence_test_tasks.yaml)).

Our Dataset class for cross-modal ablation on Flickr30K Entites is implemented in [`volta/volta/datasets/flickr30ke_ablation_dataset.py`](volta/volta/datasets/flickr30ke_ablation_dataset.py).

The LabelMatch subset can be derived following the notebook [`notebooks/Data-MakeLabelMatch.ipynb`](notebooks/Data-MakeLabelMatch.ipynb).


## Models

Most of the models we evaluated were released in [VOLTA](https://github.com/e-bug/volta) ([Bugliarello et al., 2021](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00408/107279/Multimodal-Pretraining-Unmasked-A-Meta-Analysis)).

If you are interested in using some of the variations we studied in our paper, reach out to us or open an issue on GitHub.


## Training and Evaluation

We introduce the following scripts in this repository:
- [`volta/train_bert_concap.py`](volta/train_bert_concap.py): Pretrain a text-only model on the textual modality of Conceptual Captions. We use this script to train BERT-CC in our paper.
- [`volta/train_concap_vis.py`](volta/train_concap_vis.py): Pretrain only on the visual modality of Conceptual Captions.
- [`volta/ablate_textonly_lang.py`](volta/ablate_textonly_lang.py): Evaluate the performance of text-only models in predicting the masked phrases.
- [`volta/ablate_vis4lang.py`](volta/ablate_vis4lang.py): Evaluate the performance of V&L models in predicting masked phrases as visual inputs are ablated.
- [`volta/ablate_lang4vis.py`](volta/ablate_lang4vis.py): Evaluate the performance of V&L models in predicting masked objects as textual inputs are ablated.

We provide all the scripts we used in our study under [experiments/](experiments).

We share our results aggregated in TSV files under [notebooks/](notebooks).


## License

This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data sets are subject to their respective licenses. <br>
If you find our code/data/models or ideas useful in your research, please consider citing the paper:
```
@inproceedings{frank-etal-2021-vision,
    title = "Vision-and-Language or Vision-for-Language? {O}n Cross-Modal Influence in Multimodal Transformers",
    author = "Frank, Stella and Bugliarello, Emanuele and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021)",
    month = "nov",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2109.04448",
}
```
