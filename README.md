# CausalMixNet
Confounding factors inherent in medical images can significantly impact the causal exploration capabilities of deep learning models, resulting in compromised accuracy and diminished generalization performance. In this paper, we present an innovative methodology named CausalMixNet that employs query-mixed intra-attention and key&value-mixed inter-attention to probe causal relationships between input images and labels. For mitigating unobservable confounding factors, CausalMixNet integrates the non-local reasoning module (NLRM) and the key&value-mixed inter-attention (KVMIA) to conduct a front-door adjustment strategy. Furthermore, CausalMixNet incorporates a patch-masked ranking module (PMRM) and query-mixed intra-attention (QMIA) to enhance mediator learning, thereby facilitating causal intervention. The patch mixing mechanism applied to query/(key&value) features within QMIA and KVMIA specifically targets lesion-related feature enhancement and the inference of average causal effect inference. CausalMixNet consistently outperforms existing methods, achieving superior accuracy and F1-scores across in-domain and out-of-domain scenarios on multiple datasets, with an average improvement of 3% over the closest competitor. Demonstrating robustness against noise, gender bias, and attribute bias, CausalMixNet excels in handling unobservable confounders, maintaining stable performance even in challenging conditions.

Dataset Preparation
---
BRACS: https://www.bracs.icar.cnr.it/ <br>
DDR: https://www.kaggle.com/datasets/mariaherrerot/ddrdataset <br>
APTOS: https://www.kaggle.com/competitions/aptos2019-blindness-detection <br>
FGADR: https://csyizhou.github.io/FGADR/ <br>
NIH dataset: https://www.kaggle.com/datasets/nih-chest-xrays/data <br>
CUB-200-2011: https://www.kaggle.com/datasets/wenewone/cub2002011 <br>
Please prepare the data as described in the following link: [https://www.sciencedirect.com/science/article/pii/S1361841525001288](https://www.sciencedirect.com/science/article/pii/S1361841525001288). <br>

How to run
---
For the DDR dataset:
```bash
python main_oct_ddr.py --data covid --source_domains DDR --device cuda:1 --algorithm resnet18-MIX-SP --K 5 --ratio 0.8 --alpha 0.5 --beta 1.0 --N_Times 20 --batch_size 64 --lr 0.0001
```

For the BRACS dataset:
```bash
python main_bracs.py --data he --source_domains APT --algorithm resnet18-MIX-SP --K 5 --ratio 0.8 --alpha 5.0 --beta 3.0 --N_Times 20 --device cuda:0 --batch_size 16 --lr 0.0001
python main_bracs_binary.py --data he --source_domains APT --algorithm resnet18-MIX-SP --K 5 --ratio 0.8 --alpha 5.0 --beta 3.0 --N_Times 20 --device cuda:0 --batch_size 16 --lr 0.0001
```

For the CUB dataset:
```bash
python main_cub.py --data cub --source_domains APT --device cuda:6 --algorithm resnet18-MIX-SP --K 3 --ratio 0.8 --alpha 5.0 --beta 5.0 --N_Times 0 --batch_size 128 --lr 0.001
```

For the NIH dataset:
```bash
python main_xray.py --data covid --source_domains APT --algorithm resnet18-MIX-SP --K 5 --ratio 0.8 --alpha 5.0 --beta 5.0 --N_Times 20 --device cuda:1 --batch_size 32 --lr 0.0001
```

Environment Requirements
---
PyTorch 2.10.0 + CUDA 12.6, Torchvision 0.25.0

Citation
---
@article{zhang2025causalmixnet,  <br>
  title={CausalMixNet: A mixed-attention framework for causal intervention in robust medical image diagnosis}, <br>
  author={Zhang, Yajie and Huang, Yu-An and Hu, Yao and Liu, Rui and Wu, Jibin and Huang, Zhi-An and Tan, Kay Chen}, <br>
  journal={Medical Image Analysis}, <br>
  volume={103}, <br>
  pages={103581}, <br>
  year={2025}, <br>
  publisher={Elsevier} <br>
}

Contact
---
If you have any questions, please contact rubyzhangyajie@gmail.com 
