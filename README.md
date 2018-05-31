Hooray~! my fellow KDD competitors. I entered this competition on day 1 and very quickly established a reasonable baseline. Due to some personal side of things, I practically stopped improving my solutions since the beginning of May. Even though my methods did not work really well compared to many top players in phase 2, but I think my solution may worth sharing due to it is relative simplicity. I did not touch the `meo` data at all, and one of my models is just calculating medians.

### Alternative data source
For new hourly air quality data, as shared in the forum, I am using [this](https://biendata.com/forum/view_post_category/102) for London and [this](https://biendata.com/forum/view_post_category/94) for Beijing instead of the API from the organizer.  

### Handling missing data
I filled missing values in air quality data with 3 steps:   
1. Fill missing values for a station-measure combo based on the values from other stations.  
To be specific:
I trained 131 lightgbm regressors for this. If PM2.5 reading on 2:00 May 20th is missing for Beijing aotizhongxin station, the regressor *aotizhongxin_aq-PM2.5* will predict this value based on known PM2.5 readings on 2:00 May 20th from 34 other stations in Beijing.    
I used thresholds to decide whether to do this imputation or not. If more than the threshold number of stations also don't have a reading, then skip this step.  
2. Fill the remaining missing values by looking forward and backward to find known values.
3. Finally, replace all remaining missing values by overall mean value.

### Approaches
#### 1. median of medians    
This is a simple model that worked reasonably well in [this Kaggle competition](https://www.kaggle.com/c/web-traffic-time-series-forecasting).

To predict PM2.5 reading on 2:00 May 20th for aotizhongxin, look back for a window of days history, calculating the median 2:00 PM2.5 readings from aotizhongxin in that window. You do this median calculation exercise for a bunch of different window sizes to obtain a bunch medians. The median value of those medians is used as the prediction.   

Intuitively this is just an aggregated *yesterday once more*. With more larger windows in the collection, the model memorizes the long-term trend better. The more you add in smaller windows, the quicker the model would respond to recent events.  

#### 2. facebooks' [prophet](https://github.com/facebook/prophet)  
This is practically even simpler than the median of medians. I treated the number of days history I throw at it and the model parameters `changepoint_prior_scale`, `n_changepoints`  as main hyperparameters and tweaked them. I did a bit work to parallelizing the fitting process for all the station-measure combos to speed up the fitting process, other than that, it is pretty much out of the box.   

I tried to use holiday indicator or tweaking other parameters of the model and they all degrade the performance of the model.  

#### 3. neural network
My neural network is a simple feed-forward network with a single shortcut, shamelessly copied [the structure](https://github.com/jfpuget/Kaggle/blob/master/WebTrafficPrediction/keras_simple.ipynb) from a senior colleague's Kaggle solution with tweaked hidden layer sizes.  
The model looks like this:  
 ![nn_plot](https://github.com/ryancheunggit/kddcup2018-of-fresh-air/blob/master/python/nn_model_arch.png?raw=true)  

The input to the neural network are concatenated (1) raw history readings, (2) median summary values from different window_sizes, and (3) indicator variables for the city, type of measure.  

The output layer in the network is a dense layer with 48 units, each corresponding to an hourly reading in the next 48 hours.    

The model is trained directly using smape as loss function with Adam optimizer. I tried standardizing inputs into zero mean and unit variance, but it will cause a problem when used together with smape loss, thus I tied switching to a clipped MAE loss, which produced similar results compared to raw input with smape loss.    

The model can be trained on CPU only machine in very short time.    

I tried out some CNN, RNN models but couldn't get them working better than this simple model, and had to abandon them.

### Training and validation setup
This is pretty tricky, and I am still not quite sure if I have done it correctly or not.  
#### For approach 1 and 2
I tried to generate predictions for a few historical months, calculating daily smape scores locally. Then sample 25 days out to calculate a mean smape score. Do this sample-scoring a large number of times and take mean as local validation score. I used this score to select parameters.  

#### For neural network
I split the history data into (X, y) pairs based on a splitting day, and then move the splitting day backward by 1 day to generate another (X, y) pair. Do this 60 times and vertically concatenate them to form my training data.   

I used groupedCV split on the concatenated dataset to do cross-validation so that measures from one station don't end up in both training and validation set. During training, the batch size is specified so that data in the batch all based on the same splitting day. I did this trying to preventing information leaking.

I got average smape scores 0.42~44 for Beijing and 0.31-0.33 for London in my local validation setting. Which I think is pretty aligned with how it averages in May.


### Closing   
Without utilizing any other weather information or integrating any sort of forecasts, all my models failed miserably for events like the sudden peak on May 27th in Beijing.
