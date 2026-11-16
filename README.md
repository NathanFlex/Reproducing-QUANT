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
