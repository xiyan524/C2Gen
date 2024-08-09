# Exploring Continual Learning of Compositional Generalization in NLI
Code and Data for the TACL paper (to appear) [Exploring Continual Learning of Compositional Generalization in NLI](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00680/123881)

We introduce the Continual Compositional Generalization in Inference (C2Gen NLI) challenge, where a model continuously acquires knowledge of constituting primitive inference tasks as a basis for compositional inferences. We explore how continual learning affects compositional generalization in NLI, by designing a continual learning setup for compositional NLI inference tasks.
 
## Data
#### Data Format
```
{
"verdical_label": "negative",        
"sick_label": "neutral",    
"sent1": "A man fails to make a snowball", 
"sent2": "A man plays with a ball", 
"mid_sent": "A man makes a snowball",
"label": "neutral"   
}
```

#### Data Download (OneDrive)
link: https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/fuxiyan_mail_nankai_edu_cn/ESEWwc_DriJJuYY1tMC09_4BqsmW5-JJCWCjCJTc7MkrbA?e=IUoogk

## Code
**preliminary**: split data as you require from the provided dataset

**environment**: python3.7, pytorch1.7.1

**run**: the script is provided in the run.sh

Acknowledgement: The code of continual learning strategies come from [VisCOLL](https://github.com/INK-USC/VisCOLL)

## Citations
Please cite our paper if you are using this dataset.
```
@article{fu2024exploring,
  title={Exploring continual learning of compositional generalization in NLI},
  author={Fu, Xiyan and Frank, Anette},
  journal={Transactions of the Association for Computational Linguistics},
  volume={12},
  pages={912--932},
  year={2024},
  publisher={MIT Press 255 Main Street, 9th Floor, Cambridge, Massachusetts 02142, USA~…}
}
```

