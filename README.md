This is the PyTorch implementation of HiPool: Hierarchical Pooling for Long Document Classification, published in ACL 2023.

The code is based on this implementation: https://github.com/IreneZihuiLi/HiPool/

Install the required files from requirements.txt

**Datasets:**

Datasets are available in the datasets folder in the following link:
https://drive.google.com/drive/folders/1L82q7hyoUm-w3tPXJpSmMbg5RDINiodV?usp=drive_link


**Running:**

To run the model on the Amazon-2048 dataset run the Notebook file A2048_Multi-roberta.ipynb
To run the model on the Amazon-512 dataset run the Notebook file A2048_Multi-roberta.ipynb
Note: Change the path of the datasets accordingly

**File Structure**
```bash
NLP-project/
├── A2048.ipynb                      -- contains Robustness code for the English language on the A2048 dataset
├── A2048_Multi-roberta.ipynb        -- contains Multilingualty code for all languages on A2048 dataset
├── A512.ipynb                       -- contains Robustness code for the English language on the A512 dataset
├── datasets/                        -- directory for datasets
├── graphModels.py                   -- graph model classes
├── graphModels_utils.py             -- helper functions for graph model classes
├── NLP_logBook.docx
├── inference_robustness.ipynb       -- contains code for robustness on the test set
├── inference_robustness_512.ipynb   -- contains code for robustness on the Amazon 512 test set
├── code_mixed.ipynb
├── quick_requirements.txt
├── statistical_analysis.ipynb
└── translate.ipynb
```



**Citation:**

@inproceedings{li2023hipool,
  title={HiPool: Modeling Long Documents Using Graph Neural Networks},
  author={Li, Irene and Feng, Aosong and Radev, Dragomir and Ying, Rex},
  booktitle={Proceedings of the Association for Computational Linguistics (ACL)},
  year={2023},
  url={https://arxiv.org/abs/2305.03319}
}

This Project has been done as a part of CS 678 Advanced NLP course at George Mason University under the guidance of Dr. Ziyu Yao - https://ziyuyao.org/ <br />
Contributors for this repository: <br />
Sumanth Manduru <br /> 
Pavan Kumar Reddy Katta <br />

