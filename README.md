
- NN models: [link](model/model.py): 
- Download data: 
	* [Hansen](data_scraping/web_mercator/download_quality_hansen.py)
	* [Planet](data_scraping/web_mercator/download_quality_planet.py)

- Run model:
```
	python train.py -c config.json -d <gpu_devices>
```

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
