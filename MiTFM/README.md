# MiTFM: A multi-view information fusion method based on transformer for Next Activity Prediction of Business Processes

## Usage

**To preprocess all the event logs, run: (Note that this will preprocess all the event logs, so it only needs to be run once)**

`python Read_all_log.py`

**To train, validate, and test the model, run:  (Example of event log BPI Challenge 2013 incidents)**

`python MiTFM.py --eventlog=bpi13_incidens --gpu=0`

**To load the saved model and test it on the test set, run:  (Example of event log BPI Challenge 2013 incidents)**

`python Test.py --eventlog=bpi13_incidents`

## Tools

[pytorch](https://pytorch.org/)

[python](https://www.python.org/)

## Data

The events log for the predictive busienss process monitoring can be found at [4TU Research Data](https://data.4tu.nl)
