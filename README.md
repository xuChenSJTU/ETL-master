# CUPL-master
This is a pytorch implementation of our paper: "Learning Complete User Preferences for Cross Domain Recommendation"   

**Running Tips**  
This folder contains the thhree datasets used in our paper and the codes of our CUPL model. Specifically, main_my.py is our model and the other four main_xxx.py files are four variants of CUPL in the abalation study of our paper.  

**0. Data preparation:**  
Download the processed data files from Google drive: https://drive.google.com/drive/folders/1oIABLZE0UcEylLwiJrWY5R3rw7t739Zz?usp=sharing   
and Baidu disk: https://pan.baidu.com/s/1UD5ezVqjsMBmIgQXl1CA-g  password:yk2i.  
When you download the processed_data, you can replace the dir here and directly run our codes! Note that the data_filter_amazon.py under processed_data dir is the file that pre-processes the original data. 

**1. Requirements:**  
python3.5; pytorch=1.4.0; tqdm=4.27.0; tensorboardX=1.8; pandas=0.25; numpy=1.15; networkx=2.2; logger=1.4; scipy=1.1; scikit-learn=0.20   

**2.Runing commands**  
Run the codes with the following commands on different datasets (amazon means "Movie & Book", amazon2 means "Movie & Music" and amazon3 means "Music & Book").  

-->on Movie & Book dataset: 
CUDA_VISIBLE_DEVICES=gpu_num python main_my.py --dataset=amazon --reg=5.0  

-->on Movie & Music dataset:  
CUDA_VISIBLE_DEVICES=gpu_num python main_my.py --dataset=amazon2 --reg=0.5  

-->on Music & Book dataset:  
CUDA_VISIBLE_DEVICES=gpu_num python main_my.py --dataset=amazon3 --reg=1.0  

In this way, you can get he results. Besides, if you want to run the variants of CUPL, just follow the same way while with different main files.  

This paper is submitted to a journal and we would release more information later.
