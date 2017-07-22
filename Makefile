setup:
	bash setup.sh

train:
	python train.py

export_model:
	python ./tools/freeze_graph.py  --model_folder=${model_folder} --net_name=${net_name}

demo:
	python app.py
