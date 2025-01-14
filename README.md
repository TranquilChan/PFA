# 
<div align="center">
  <p align="center">
    <img src="./docs/logo.png" align="middle" width = "1000" />
  </p>

  ***Scribble-Supervised Semantic Segmentation with Prototype-based Feature Augmentation***


</div>


## <img src="docs/news.png" width="30"/> News

*  [2024-05-02] 🔥 The paper has been accepted by ICML 2024.


##  <img src="docs/intro.png" width="30"/> Abstract

Scribble-supervised semantic segmentation presents a cost-effective training method that utilizes annotations generated through scribbling. It is valued in attaining high performance while minimizing annotation costs, which has made it highly regarded among researchers. Scribble supervision propagates information from labeled pixels to the surrounding unlabeled pixels, enabling semantic segmentation for the entire image. However, existing methods often ignore the features of classified pixels during feature propagation. To address these limitations, this paper proposes a prototype-based feature augmentation method that leverages feature prototypes to augment scribble supervision. Experimental results demonstrate that our approach achieves state-of-the-art performance on the PASCAL VOC 2012 dataset in scribble-supervised semantic segmentation tasks.


## <img src="./docs/structure.png" width="30"/>  Code structure

``` plain
├── config                          
│     └── voc.yaml                   
│  
├── datasets                           
│     └── voc2012                
│           ├── ImageSets  
│           ├── JPEGImages 
│           ├── ScribbleLabels        
│           └── cls_labels.npy     
│       
├── model                           
│     ├── basic_model.py                  
│     ├── mix_tr.py             
│     └── seg_head.py                  
│ 
├── utils                            
│     ├── dataset.py               
│     ├── loss.py               
│     ├── tools.py                 
│     ├── trains.py             
│     └── utils.py                
│ 
├── weights                         
│     ├── checkpoints
│     │     ├── ....             
│     │     └── ....                            
│     └── pretrained                
│           ├── mit_b1.pth             
│           └── ....               
└── train.py                           
```
## <img src="./docs/data.png" width="30"/>  Preparations


* ### Dataset:

The dataset can be downloaded on [Link](https://jifengdai.org/downloads/scribble_sup/).

* ### Pre-trained Weight:

The Pre-trained weight can be downloaded on [Link](https://github.com/NVlabs/SegFormer).

## <img src="./docs/train.png" width="30"/>  Quick start

```shell
python train.py
```

## <img src="./docs/citation.png" width="30"/>  Citation

```latex
@inproceedings{Chan2024Scribble,
  title={Scribble-Supervised Semantic Segmentation with Prototype-based Feature Augmentation},
  author={Guiyang Chan and Pengcheng Zhang and Hai Dong and Shunhui Ji and Bainian Chen},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}

```

## <img src="./docs/love.png" width="30"/>  Acknowledgement

We use [SegFormer](https://github.com/NVlabs/SegFormer) as the backbone.
Thank them for their excellent work.

