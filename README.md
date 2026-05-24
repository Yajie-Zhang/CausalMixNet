# CausalMixNet
The source code of CausalMixNet

Dataset Preparation
---
Please prepare the data as described in the following link: [https://www.sciencedirect.com/science/article/pii/S1361841525001288](https://www.sciencedirect.com/science/article/pii/S1361841525001288). <br>

How to run
---
python main.py --data he --source_domains APT --device cuda:0 --algorithm resnet18-MIX-SP --K 5 --ratio 0.8 --alpha 5.0 --beta 3.0 --N_Times 20 --batch_size 16

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
