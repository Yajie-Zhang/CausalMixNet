# CausalMixNet
Confounding factors inherent in medical images can significantly impact the causal exploration capabilities of deep learning models, resulting in compromised accuracy and diminished generalization performance. In this paper, we present an innovative methodology named CausalMixNet that employs query-mixed intra-attention and key&value-mixed inter-attention to probe causal relationships between input images and labels. For mitigating unobservable confounding factors, CausalMixNet integrates the non-local reasoning module (NLRM) and the key&value-mixed inter-attention (KVMIA) to conduct a front-door adjustment strategy. Furthermore, CausalMixNet incorporates a patch-masked ranking module (PMRM) and query-mixed intra-attention (QMIA) to enhance mediator learning, thereby facilitating causal intervention. The patch mixing mechanism applied to query/(key&value) features within QMIA and KVMIA specifically targets lesion-related feature enhancement and the inference of average causal effect inference. CausalMixNet consistently outperforms existing methods, achieving superior accuracy and F1-scores across in-domain and outof-domain scenarios on multiple datasets, with an average improvement of 3% over the closest competitor. Demonstrating robustness against noise, gender bias, and attribute bias, CausalMixNet excels in handling unobservable confounders, maintaining stable performance even in challenging conditions.

Dataset Preparation
---
Please prepare the data as described in the following link: [https://www.sciencedirect.com/science/article/pii/S1361841525001288](https://www.sciencedirect.com/science/article/pii/S1361841525001288). <br>

How to run
---
python main.py --data dataset_name --device cuda:0 --algorithm resnet18-MIX-SP --K 5 --alpha ALPHA --beta BETA --batch_size BS --lr LR

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
