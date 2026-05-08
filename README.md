# AML

Project for the Master Degree course of Advanced Machine Learning at University of Verona lectured by Dr. Yiming Wang

## REQUIREMENTS

### REFERENCES

- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [PyTorch Releases](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)

### PACKAGES

```bash
conda env create -n aml python=3.10
```

```bash
conda install --yes cudatoolkit=12.1
```

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

```bash
pip install ftfy regex tqdm
```

```bash
pip install git+https://github.com/openai/CLIP.git
```

## ROADMAP

1. Implement and use the original CLIP model from OpenAI

2. Test the model on the dataset provided by the AML competition

3. Implement and test the fine-tuning proposed in [Fully Fine-tuned CLIP models](https://arxiv.org/pdf/2407.04003)

4. Here what Elia proposes as "novel": if it works and we see an improvement on results, then we can proceed to do data augmentation with SAM

5. Now we can use the augmented dataset and repeat the process of 3. with the new dataset

6. At this point, we can evaluate different approaches if there are like Tip-Adapter

7. In the end, we can write the report
