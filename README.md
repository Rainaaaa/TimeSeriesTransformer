# TimeSeriesTransformer
This Time-series based Transformer is implemented with monthly individual stock returns from March 1957 to December 2016. The data is from the Center for Research in Securities Prices (CRSP) for all firms listed in the three major exchanges (NYSE, AMEX, and NASDAQ).Based on each month's return(r) and characteristics(z), we calcuated r0-r6 and z_i_0 - z_i_6 by matching realized returns at month t with the most recent monthly return and characteristics at the end of month t-1 to t-6. Our goal is to predict r6.

- v1 - Simultaneous Return Transformer
- v1.3 - Characteristics return Transformer with one Characteristic only
- v1.4 - Characteristics return Transformer with all Characteristics; Processing with IO streaming(column by column);

## folders (v1.4)
logs: contains all the log files (all the printed result will be written in log files)

output: contains all the jobs log files running on cluster

tools: bash scripts for running scripts on cluster

checkpoints: stored the trained model and model checkpoints

processed data: Contains all the processed data and the data loaders



## Scripts (v1.4)
logger.py = for generating log files

utils.py = conatins a list of characteristics names

dataset.py = contains all the data processing functions 

test_processing.py = contains the testing phase of the processing part (run this part to obatined the processed data)

test_training.py = contains the testing phase of the training part (run this part to train the model)

test_testing.py = contains the testing phase of the testing part (run this part to test the trained model and obatin the Sharp Ratio)

modules2.py = contains the transformer code (contains the encoder)
- encoder: r0-r3, z0-z3
- decoder: r4-r5, z4-z5
- output: r5-r6 (r6 is the result we want to predict)


**note:** Ignore the encoder.py

