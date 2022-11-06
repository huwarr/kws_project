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

`code_final.ipynb` - code, comments and graphics for both parts of the homework. Includes:

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

