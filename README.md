# CMIFN

## Discriminative analysis of features
To overcome the challenge of space limitations, we have provided a visualization analysis using t-SNE to demonstrate the discriminative power of our model's features on Weibo dataset. In this figure, we notice that the identified features exhibit distinct separability, indicating that they can be clearly distinguished from one another in the context of our model's classification process.
![tsne](https://github.com/12y3/CMIFNcode/assets/87634436/3b66a49d-eea9-4119-9b96-83b5f2bf1b64)


## Results on Fakeddit dataset
Multimodal fake news detection is a very important topic because fake news has a profound impact on people's lives. To make a fair comparison in this field, we selected the two most commonly used datasets in fake news detection research - Weibo and Twitter. These datasets are not only widely considered standard, but also provide rich and diverse real-world examples. Additionally, we have conducted experiments on the Fakeddit dataset as well and observed significant improvements in our method over other techniques. Due to space constraints, these additional experimental results were not included in the paper, but we have provided detailed results in the online appendix.

In the Fakeddit dataset[7], we randomly select 30,000 image-text pairs in the Fakeddit training set as our training set and randomly select 10,000 image-text pairs from the test set as our test set. 
From the table, we can observe that our method's accuracy surpasses the CAFE method by 0.022 and the MRML method by 0.094, demonstrating the superiority of our approach.

![Fake](https://github.com/12y3/CMIFNcode/assets/87634436/47f34804-aff8-49aa-8d50-08e3a6ab44ee)


[1] Y. Wang, F. Ma, Z. Jin, Y. Yuan, G. Xun, K. Jha, L. Su, and J. Gao, “Eann: Event adversarial neural networks for multi-modal fake news detection,” in ACM Sigkdd International Conference on Knowledge Discovery & Data Mining, 2018, pp. 849–857.   
[2] S. Singhal, R.R. Shah, T. Chakraborty, P. Kumaraguru, and S. Satoh, “Spotfake: A multi-modal framework for fake news detection,” in International Conference on Multimedia Big Data, 2019, pp. 39–47.  
[3] T. Zhang, D. Wang, H. Chen, Z. Zeng, W. Guo, C. Miao, and L. Cui, “Bdann: Bert-based domain adaptation neural network for multi-modal fake news detection,” in International Joint Conference on Neural Networks, 2020, pp. 1–8.  
[4] P. Wei, F. Wu, Y. Sun, H. Zhou, and X. Jing, “Modality and event adversarial networks for multi-modal fake news detection,” IEEE Signal Processing Letters, vol. 29, pp. 1382–1386, 2022.   
[5] Y. Chen, D. Li, P. Zhang, J. Sui, Q. Lv, L. Tun, and L. Shang, “Cross-modal ambiguity learning for multimodal fake news detection,” in ACM Web Conference, 2022, pp. 2897–2905.  
[6] L. Peng, S. Jian, D. Li, and S. Shen, “Mrml: Multimodal rumor detection by deep metric learning,” in IEEE International Conference on Acoustics, Speech and Signal Processing, 2023, pp. 1–5.  
[7] Nakamura K, Levy S, Wang W Y. r/fakeddit: A new multimodal benchmark dataset for fine-grained fake news detection[J]. arXiv preprint arXiv:1911.03854, 2019.

