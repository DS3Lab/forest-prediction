
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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|   |   |-- five_pct/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|   |   |   |-- 2018/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|   |   |   |   |-- *.png  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|   |   |   |-- 2017/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|   |   |   |-- ...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|   |   |-- four_pct/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|   |   |   |-- 2018/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|   |   |   |-- 2017/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|   |   |   |-- ...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|   |-- planet/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|   |   |   |-- *.png  
