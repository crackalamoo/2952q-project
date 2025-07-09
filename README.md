# Universal Adversarial Trigger Algorithm for Protein Folding

In natural language processing, many large language models show greatly reduced performance on a prompt if the prompt is preceded or followed by particular triggers. Given access to the model, these triggers can be discovered through a white-box search. In this paper, we extend the algorithm for discovering such triggers from natural language processing to the domain of protein folding. We run the algorithm in an adversarial setting to increase the model’s error, as well as the reverse to discover triggers that decrease the model’s error. We also repurpose this algorithm for the problem of inverse folding: predicting a protein’s sequence given its structure. While the adversarial and inverse folding settings were unsuccessful, we discovered a trigger that spuriously increases prediction accuracy in ESMFold despite making the sequence less similar from the ground truth. We suggest a number of applications and future directions based on this research.

[PDF](./2952Q_project.pdf)

[Slides (12/2/24)](https://docs.google.com/presentation/d/10df4CHvSGdoUeD3ukY5nUuqw71NGTgM_xWRiHb8qUJw/edit?usp=sharing)