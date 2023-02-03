# Value Zeroing
The official repo for the [EACL 2023](https://2023.eacl.org/) paper "__Quantifying Context Mixing in Transformers__"

📃[[Paper]](https://arxiv.org/pdf/2301.12971.pdf)

🤗[[Gradio Demo]](https://huggingface.co/spaces/amsterdamNLP/value-zeroing)

## Abstract
> Self-attention weights and their transformed variants have been the main source of information for analyzing token-to-token interactions in Transformer-based models. But despite their ease of interpretation, these weights are not faithful to the models’ decisions as they are only one part of an encoder, and other components in the encoder layer can have considerable impact on information mixing in the output representations. In this work, by expanding the scope of analysis to the whole encoder block, we propose _Value Zeroing_, a novel context mixing score customized for Transformers that provides us with a deeper understanding of how information is mixed at each encoder layer. We demonstrate the superiority of our context mixing score over other analysis methods through a series of complementary evaluations with different viewpoints based on linguistically informed rationales, probing, and faithfulness analysis.



## External links 

### Models, Data and Preprocessing Toolkits
* [HuggingFace’s Transformers](https://github.com/huggingface/transformers)
* [BLiMP](https://github.com/alexwarstadt/blimp)
* [spaCy](https://github.com/explosion/spaCy)
* [NeuralCoref](https://github.com/huggingface/neuralcoref)

### Baselines
* [Attention Rollout](https://github.com/samiraabnar/attention_flow)
* [Attention-norm](https://github.com/gorokoba560/norm-analysis-of-transformer)
* [GlobEnc](https://github.com/mohsenfayyaz/GlobEnc)
* [ALTI](https://github.com/mt-upc/transformer-contributions)
* [Captum](https://captum.ai/) for Gradient-based methods
