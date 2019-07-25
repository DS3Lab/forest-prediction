
NN models: https://github.com/DS3Lab/forest-prediction/blob/master/model/model.py
Download data: 
	https://github.com/DS3Lab/forest-prediction/blob/master/data_scraping/web_mercator/download_quality_planet.py
	https://github.com/DS3Lab/forest-prediction/blob/master/data_scraping/web_mercator/download_quality_hansen.py

Run model:
	python train.py -c config.json -d <gpu_devices>

Dataset format:
        Project/
        |-- min_quality/
        |   |-- hansen/
        |   |   |-- five_pct/
        |   |   |   |-- 2018/
        |   |   |   |   |-- *.png
        |   |   |   |-- 2017/
        |   |   |   |-- ...
        |   |   |-- four_pct/
        |   |   |   |-- 2018/
        |   |   |   |-- 2017/
        |   |   |   |-- ...
        |   |-- planet/
        |   |   |   |-- *.png