# Backdoor Attack on Diffusion Model

## Project Overview
This project focuses on implementing a backdoor attack on diffusion models, exploring security vulnerabilities in generative AI. The attack modifies the training dataset with a stealthy trigger to influence generated outputs while remaining undetectable. This research was conducted as part of the *Secure Design for Machine Learning* course at the Illinois Institute of Technology.

## Features
- **Backdoor Attack Implementation**: Introduces a malicious trigger into a diffusion model's training dataset.
- **Data Poisoning Technique**: Adjusts the poisoning rate to balance stealth and effectiveness.
- **Diffusion Model Architecture**: Encoder-decoder structure using Gaussian noise for data generation.
- **PixelCNN Sampling**: Implements a probabilistic model for generating outputs.
- **GPU Optimization**: Provides solutions for CUDA memory errors when running on GPUs.

## Technologies Used
- **Programming Language**: Python
- **Machine Learning Framework**: PyTorch
- **Deep Learning Architecture**: Diffusion Models, PixelCNN
- **Dataset**: MNIST (for initial testing)

## Installation
### Requirements:
```bash
Python 3.x
```

### Steps:
```bash
# Clone this repository
git clone https://github.com/gabrielpecoraro/Backdoor_Diffusion.git

```

## Usage
```bash
# Train a poisoned diffusion model
python training.py 
```

## Conducted Tests
- **Dataset**: The model was tested using the **MNIST dataset** with a poisoning rate of **0.2**.
- **Performance Comparison**: Clean and poisoned models showed similar reconstruction accuracy, proving the stealthiness of the attack.
- **Challenges**: PyTorch security checks and CUDA memory errors required workarounds for execution.

## Future Improvements
- Extend the backdoor attack to the **CIFAR-10 dataset** for real-world image impact analysis.
- Optimize **hyperparameters** to fine-tune stealth and effectiveness.
- Resolve **PyTorch/CUDA security issues** to enhance the stability of the attack implementation.

## Contributors
- **Gabriel Pecoraro**

## References
1. "Diffusion Models from Scratch" - [Michael Wornow](https://michaelwornow.net/2023/07/01/diffusion-models-from-scratch)
2. "Denoising Diffusion Probabilistic Models" - [Ho et al., 2020](https://arxiv.org/pdf/2006.11239)
3. "Pixel Recurrent Neural Networks" - [arXiv](https://arxiv.org/pdf/1601.06759)
4. "How to Backdoor Diffusion Models?" - [arXiv](https://arxiv.org/pdf/2212.05400)
5. "PyTorch Security" - [GitHub](https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---
This project demonstrates the vulnerabilities of diffusion models and highlights the risks associated with backdoor attacks in generative AI. Contributions and improvements are welcome!

