# Eyes Show the Way: Modelling Gaze Behaviour for Hallucination Detection

This repository contains the code for the EMNLP 2023 paper: [Eyes Show the Way: Modelling Gaze Behaviour for Hallucination Detection](https://aclanthology.org/2023.findings-emnlp.764.pdf) 

We collect and introduce an eye tracking corpus (IITB-HGC: IITB-Hallucination Gaze corpus) consisting of 500 instances, annotated by five annotators for hallucination detection. This dataset is available on huggingface: [IITB-HGC](https://huggingface.co/datasets/cfilt/IITB-HGC)

## Introduction 

Detecting hallucinations in natural language processing (NLP) is a critical undertaking that demands a deep understanding of both the semantic and pragmatic aspects of languages. Cognitive approaches that leverage users’ behavioural signals, such as gaze, have demonstrated effectiveness in addressing NLP tasks with similar linguistic complexities. However, their potential in the context of hallucination detection remains largely unexplored. 

In this paper, we propose a novel cognitive approach for hallucination detection that leverages gaze signals from humans. We first collect and introduce an eye tracking corpus (IITB-HGC: IITB-Hallucination Gaze corpus) consisting of 500 instances, annotated by five annotators for hallucination detection. 

Our analysis reveals that humans selectively attend to relevant parts of the text based on distributional similarity, similar to the attention bias phenomenon in psychology. We identify two attention strategies employed by humans: global attention, which focuses on the most informative sentence, and local attention, which focuses on important words within a sentence. Leveraging these insights, we propose a novel cognitive framework for hallucination detection that incorporates these attention biases. Experimental evaluations on the FactCC dataset demonstrate the efficacy of our approach, obtaining a balanced accuracy of 87.1%. Our study highlights the potential of gaze-based approaches in addressing the task of hallucination detection and sheds light on the cognitive processes employed by humans in identifying inconsistencies. 

The following image shows an overview of the end-to-end model incorporating global attention and local attention bias for detecting hallucination given a claim and a context:
![image](https://github.com/kishanmaharaj/gaze-hallucination-detection/assets/16451688/492de5cb-453e-4b5c-bd55-15e74042f242)

## Cite the work

If you make use of the dataset or the code please cite our paper.

```
@inproceedings{maharaj2023eyes,
  title={Eyes Show the Way: Modelling Gaze Behaviour for Hallucination Detection},
  author={Maharaj, Kishan and Saxena, Ashita and Kumar, Raja and Mishra, Abhijit and Bhattacharyya, Pushpak},
  booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023}
}
```
