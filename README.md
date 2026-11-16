# Scale-Aware QUANT

***Reproducing QUANT: Scale-Aware Feature Allocation for Efficient Time Series Classification***

## Results
* [Mean Accuracy, Balanced Accuracy, Log Loss and AUROC + Training/Test Times](./results/scale_aware_results.csv)
* [Per Resample](./results/scale_aware_resamples_results.csv)

## Requirements

* Python
* PyTorch
* NumPy
* scikit-learn (or similar)

## Code
### [`quant_scale_aware.py`](./code/quant_scale_aware.py)

## Usage

### Obtaining all UCR142 datasets
* Download the original 112 folds from [here](https://drive.google.com/file/d/1V36LSZLAK6FIYRfPx6mmE5euzogcXS83/view)
* Download the new 30 from [here](https://drive.google.com/file/d/1vuh6mgNrNKjHr9MMRQP0J0_gGA4dE7E3/view)
* Download the resample indices from [here](https://drive.google.com/file/d/1aLBP_nhnoqz075puKg30zuF3F_QBOXYM/view)

### Running the experiments
Please refer [run.py](./code/run.py) for how to reproduce our results

## Alternative Approaches
We attempted to reduce dependence on the `div` hyper parameter entirely by scaling the number of quantiles based on the length of the intervals at each level. 
[Square root scaled num_quantiles](./results/sqrt_scaled_mean_results.csv)
The test error for the `Phoneme` series was found to be extremely low. Random samples of the 5 folds with the least accuracy can be visualized through the notebook [visualize_series.ipynb](./code/visualize_series.ipynb). In an attempt to remedy this we extended the initial 4 representation features to 6, by adding a rolling skewness measure and a rolling Zero-Crossing Rate measure.
[6 Representation features (Rolling Skewness + Rolling ZCF)](./results/scale_aware_6_features_mean_results.csv)
