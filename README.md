# "Enhancing De Novo Drug Design Across Multiple Therapeutic Targets with CVAE Generative Models"

Authors: Romanelli Virgilio, Annunziata Daniela, Cerchia Carmen, Cerciello Donato, Piccialli Francesco, Lavecchia Antonio

Code Authors: Annunziata Daniela, Cerciello Donato, Piccialli Francesco

Contact: daniela.annunziata@unina.it, cerciellodonato@gmail.com, francesco.piccialli@unina.it

This directory contains implementations of the Conditional Variational Autoencoders (CVAE) framework for molecule generation using different grammars 
(SMILES or SELFIES), targeting Cyclin-Dependent Kinase 2 (CDK2), Peroxisome Proliferator-Activated Receptor Gamma (PPAR-γ) and Dipeptidyl Peptidase-4
(DDP-IV).

### Abstract
Drug discovery is a costly and time-consuming process, necessitating innovative strategies to enhance efficiency across different stages, from initial hit identification to final market approval. Recent advancement in deep learning (DL), particularly in de novo drug design, show promise. Generative models, a subclass of DL algorithms, have significantly accelerated the de novo drug design process by exploring vast areas of chemical space. Here, we introduce a Conditional Variational Autoencoder (CVAE) generative model tailored for de novo molecular design tasks, utilizing both SMILES and SELFIES as molecular representations. Our computational framework successfully generates molecules with specific property profiles validated though metrics such as uniqueness, validity, novelty, quantitative estimate of drug-likeness (QED), and synthetic accessibility (SA). We evaluated our model's efficacy in generating novel molecules capable of binding to three therapeutic molecular targets: CDK2, PPAR-γ and DPP-IV. Comparing with state-of-the-art frameworks demonstrated our model's ability to achieve higher structural diversity while maintaining the molecular properties ranges observed in the training set molecules. This proposed model stands as a valuable resource for advancing de novo molecular design capabilities. 


![Alt text](/cvae_architecture.jpg)
**Figure**. CVAE Architecture for molecular generation.


### Create the Environment
In the terminal, run the following command to create the environment using the provided cvae_environment.yml file:

conda env create -f cvae_environment.yml (or mamba env create -f cvae_environment.yml)

This will set up the necessary environment with all dependencies required for the project.

### Files' description
"molecules_generator.py" is responsible for generating molecules in relation to the desired targets. This script leverages the Conditional Variational Autoencoder 
(CVAE) framework to produce molecular structures using specific grammars tailored to previous targets. The generated molecules are aligned with the predefined 
target properties, enabling focused molecular design.

"analysis.py" is dedicated to the analysis of the generated molecules. This script examines the chemical properties of the molecules, including 
their structural validity, drug-likeness, and other relevant chemical metrics.

### Command molecules_generator.py:

-   grammar: Specifies the grammar, either 'smiles' or 'selfies' (default: smiles).
-   model_type: Determines whether to generate using the 'initial' or 'tuning' dataset (default: tuning).
-   target: Specifies the target for tuning: cdk2, ddp4, gamma (default: cdk2).
-   seq_length: seq_length: Sets the maximum length of the generated molecules (default: 120).
-   num_trials: The number of molecules generated per step (default: 3000).
-   num_iterations: The total number of generating steps (default: 10).

### Example command

```shell
$ python3 molecules_generator.py --grammar selfies --model_type tuning --target gamma
--seq_length 120 --num_trials 2000 --num_iterations 3
```

### Command analysis.py:

-   grammar: Specifies the grammar, either 'smiles' or 'selfies' (default: smiles).
-   target: Specifies the target analysis: cdk2, ddp4, gamma, initial (default: cdk2).


### Example command

```shell
$ python3 analysis.py --grammar selfies --target gamma
```
