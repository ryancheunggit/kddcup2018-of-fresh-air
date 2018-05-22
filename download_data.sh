#!/bin/bash
# Ren Zhang @ ryanzjlib dot gmail dot com

# if on mac use the following to install wget
# curl -O http://ftp.gnu.org/gnu/wget/wget-1.15.tar.gz
# tar -zxvf wget-1.15.tar.gz
# cd wget-1.15/
# ./configure
# note zsh may need use '' to escape on urls

# create data directory
mkdir -p features
mkdir -p input
mkdir -p models
mkdir -p submission
mkdir -p summaries
wget -nc 'https://www.dropbox.com/s/nuy1r6psk46vsi4/London_AirQuality_Stations.csv?dl=1' -O ./input/London_AirQuality_Stations.csv
wget -nc 'https://www.dropbox.com/s/pb18hqtxo2vnino/London_grid_weather_station.csv?dl=1' -O ./input/London_Grid_Stations.csv
wget -nc 'https://www.dropbox.com/s/mtyg1kitlt5k6h7/Beijing_grid_weather_station.csv?dl=1' -O ./input/Beijing_Grid_Stations.csv
wget -nc 'https://www.dropbox.com/s/464rp6lhjgu0jv6/beijing_17_18_aq.csv?dl=1' -O ./input/beijing_17_18_aq.csv
wget -nc 'https://www.dropbox.com/s/lv2i6tictta9pfq/beijing_201802_201803_aq.csv?dl=1' -O ./input/beijing_18_aq.csv
wget -nc 'https://www.dropbox.com/s/ht3yzx58orxw179/London_historical_aqi_forecast_stations_20180331.csv?dl=1' -O ./input/london_17_18_aq.csv
wget -nc 'https://www.dropbox.com/s/jwy7mdcmjes61fz/London_historical_aqi_other_stations_20180331.csv?dl=1' -O ./input/london_17_18_aq_bonus.csv
wget -nc 'https://www.dropbox.com/s/jjta4addnyjndd8/beijing_17_18_meo.csv?dl=1' -O ./input/beijing_17_18_meo.csv
wget -nc 'https://www.dropbox.com/s/94llgcr81u2tbg1/Beijing_historical_meo_grid.csv?dl=1' -O ./input/beijing_history_meo.csv
wget -nc 'https://www.dropbox.com/s/yraf9y89nhxzptd/London_historical_meo_grid.csv?dl=1' -O ./input/london_history_meo.csv
wget -nc 'https://www.dropbox.com/s/uwz2269deyxcip5/sample_submission.csv?dl=1' -O ./input/sample_submission.csv

cd python
python process_history_data.py
python external_data_download.py