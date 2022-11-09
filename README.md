### Dataset
We are using the Movie Lens Small Dataset. This dataset is retrived from <https://grouplens.org/datasets/movielens/> and is preprocessed with the file `ml-latest-small.py`.

### Experiments
We modified the original lightGCN to incorporate the movies' rating by the users, as well as the movies' genre. We call this new model "Modified LightGCN". 
We also implemented a few baselines (SVD, SVDpp and KNN) taking into account the movie ratings. The baselines selected for experiment include KNN-based models with different similarity measures (Cosine Similarity, Pearson Similarity, Pearson-Baseline, and Mean Squared Difference), SVD and SVDpp.
We perform experiments comparing the original LightGCN, Modified LightGCN and baselines on the Movie Lens Dataset. 

### To Run the Codes
#### LightGCN
`python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="ml-latest-small" --topks="[20]" --recdim=64 --testbatch 61 --epochs 500`

For training with ratings (number of edges proportional to the rating), use:
`--multiplied`

For training with movie categories, use:
`--genre`

#### Baselines
The process for executing the experiments have been encapsulated in `baselines.py` and `baseline_experiments.py`. To run the experiments, simply run `baseline_experiments.py` with the following configuration:
`--dataset="ml-latest-small" --topks="[20]" --recdim=64 --testbatch 12`

The experiment result will be written into the directory `/data/baseline_results_ML`. 

For details on the baseline experiments (e.g. trainable parameters), please refer to the comments in `baselines.py` and `baseline_experiments.py`.
