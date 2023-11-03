# HiPool

This is the pyTorch implementation of ([url](https://arxiv.org/abs/2305.03319))**, published in ACL 2023. 

The code is based on this implementation:
https://github.com/helmy-elrais/RoBERT_Recurrence_over_BERT/blob/master/train.ipynb

https://github.com/IreneZihuiLi/HiPool/


<!-- Create environment --> -->
create a pytorch virtual enivronment and install all the dependencies using:
pip install -r requirements.txt


<!-- Command to run the script -->
python train_imdb.py --sentlen 50 --adj_method bigbird --level sent --graph_type gat --epoch10 --dataset us-consumer-finance-complaints/consumer_complaints.csv
Dataset: consumer_complaints

<!-- Files -->
'train: main function.

Dataset_Split_Class.py: This file loads the data and splits the data

Bert_Classification.py: This file contains modeling for BERT, graphs, Transformers and ROBERT model implementation

Graph_Models.py: graph model classes.

requirements.txt: Contains all the dependencies

<!-- site -->
@inproceedings{li2023hipool,
  title={HiPool: Modeling Long Documents Using Graph Neural Networks},
  author={Li, Irene and Feng, Aosong and Radev, Dragomir and Ying, Rex},
  booktitle={Proceedings of the Association for Computational Linguistics (ACL)},
  year={2023},
  url={https://arxiv.org/abs/2305.03319}
}


