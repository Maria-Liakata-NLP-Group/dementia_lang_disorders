# Reformulated NLP Tasks for Capturing Language Disorders in Dementia

## This repository contains the implementation code for the paper titled "Reformulating NLP Tasks to Capture the Longitudinal Manifestation of Language Disorders in Individuals with Dementia."


## Table of Contents

- [Data](#data)
- [Usage](#usage)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

## Data
This work utilizes the Pit and ADReSS dementia corpora. Access to the data is password-protected and restricted to individuals who have signed an agreement. For additional information, please visit the DementiaBank website.


## Usage

### Run the models

1. Standard-finetune: `python train.py --model_name RoBERTa`
2. Multitask MLM: `python train.py --model_name RoBERTa_Multitask`
3. Entailment: `python train.py --model_name RoBERTa_entail`
4. Standard−prompt: `python train_prompt.py --model_name RoBERTa_Prompt`
5. Prompt−demonstrations: `python train_prompt.py --model_name RoBERTa_Prompt_dem`
6. Prompt−inverse: `python train_prompt.py --model_name RoBERTa_Prompt_inverse`

### Test the models

1. Standard-finetune: `python test.py --model_name RoBERTa`
2. Multitask MLM: `python test.py --model_name RoBERTa_Multitask`
3. Entailment: `python test.py --model_name RoBERTa_entail`
4. Standard−prompt: `python test_prompt.py --model_name RoBERTa_Prompt`
5. Prompt−demonstrations: `python test_prompt.py --model_name RoBERTa_Prompt_dem`
6. Prompt−inverse: `python test_prompt.py --model_name RoBERTa_Prompt_inverse`

## Contributing
1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit them: `git commit -m 'Add your feature`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request



## Author
Dimitris Gkoumas. For more information, please visit [gkoumasd.github.io](https://gkoumasd.github.io)  


## License
If you find this project useful for your research, please consider citing it using the following BibTeX entry:


```bibtex
@inproceedings{gkoumas2023reformulating,
  title={Reformulating NLP tasks to Capture Longitudinal Manifestation of Language Disorders in People with Dementia.},
  author={Gkoumas, Dimitris and Purver, Matthew and Liakata, Maria},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={15904--15917},
  year={2023}
}