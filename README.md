
- NN models: [link](model/model.py): 
- Download data: 
	* [Hansen](data_scraping/web_mercator/download_quality_hansen.py)
	* [Planet](data_scraping/web_mercator/download_quality_planet.py)

- Run model:
```
	python train.py -c config.json -d <gpu_devices>
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
- Dataset format:
&nbsp;&nbsp;Project/
&nbsp;&nbsp;&nbsp;&nbsp;|-- min_quality/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;|-- hansen/  
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
