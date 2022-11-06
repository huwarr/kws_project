# Homework 2 (KWS)

### Requirements

First, clone this repository and install all the necessary requirements by running:

```bash
git clone https://github.com/huwarr/kws_project.git
```

```bash
pip install -r requirements.txt
```

### Checkpoints

All the necessary checkpoints are situated in this repository. Alternatively, they may be downloaded with running a `setup.sh` script:

```bash
sh setup.sh
```

### Repository structure

- `code_final.ipynb` - code, comments and graphics for both parts of the homework. Includes:

  - A copy of `seminar.ipynb` for training the base model.
    Quality of the base model: `4.58e-5`

  - **Streaming KWS** - implementation of streaming KWS and an example of applying it to the audio track.

  - **Speed up & Compression** - an implementation of 3 approaches to compress and speed up the base model.
      1. **Dark Knowledge Distilation**
      
         Distilled a smaller model with the base model as the teacher.

         Qulity of student model: `5.49e-5 < 5.5e-5 * 1.1` as was required.

         ```bash
         compression rate: 10.2823
         speed up rate: 10.9215
         ```

         ***I would like this approach to be accounted for when grading my homework.***

      2. **Quantization**
      
         Applied [Post Training Quantization](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#post-training-static-quantization) to the base model.

         Quality: `3.93e-5`

         ```bash
         compression rate: 2.7542
         speed up rate: 2.7313
         ```

       3. **Dark Knowledge Distillation + Quantization**
       
          Applied Post Training Quantization to the distilled version of the base model.

          Quality: `6.25e-5`

          ```bash
          compresion rate: 11.2051
          speed up rate: 17.8776
          ```
- `base.pt` - state dict of the trained base model

- `streaming_kws.pth` - streaming KWS saved in `torch.jit` format
- `stream.py` - a function to run streaming KWS with obtaining audio via microphone

- `distilation_jit.pth` - a distilled base model, saved in `torch.jit` format
- `distilation_state_dict.pt` - state dict of the distilled base model

- `requirements.txt` - all the necessary requirements for the code, provided in `code_final.ipynb`

- `setup.sh` - a bash script to download all checkpoints (`.pt` and `.pth`  files)


### Sources

1. [Streaming Aware neural network models: README](https://github.com/google-research/google-research/blob/master/kws_streaming/README.md)
2. [Post Training Quantization: pytorch tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#post-training-static-quantization)
3. [Distilling the Knowledge in a Neural Network: paper](https://arxiv.org/pdf/1503.02531.pdf)
4. [Saving and Loading Models: pytorch tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
