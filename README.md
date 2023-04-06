# anomaly_detection_time_series
The content of the repository is organized as show below:

 

├ configuration

              |- conv_ae.yaml

              |- conv_ae1D.yaml

              |- lstm.yaml

              |- lstm_ae.yaml

              |- lstm_vae.yaml

              |- lstm_vae1cell.yaml

├ dataset

              |- sentinel.py

├ esa

              |- conv_utils.py

	      |- load_utils.py

	      |- lstm_utils.py

	      |- plot_results_conv_ae.ipynb

	      |- plot_results_conv_ae1D.ipynb

	      |- plot_results_lstm_esa.ipynb

	      |- plot_results_lstm_forecast.ipynb

	      |- plot_results_vae_lstm.ipynb

├ models

              |- conv_ae.py

              |- conva_ae_1D.py

              |- lstm.py

              |- lstm_ae.py

              |- lstm_vae.py

              |-lstm_vae1cell.py

├ notebook

              |- notebook_utils.py

	      |- plot_results_conv_ae.ipynb

	      |- plot_results_conv_ae1D.ipynb

	      |- plot_results_lstm.ipynb

	      |- plot_results_lstm_ae.ipynb

	      |- plot_results_lstm_vae.ipynb

	      |- plot_results_lstm_vae1cell.ipynb

 
|- utils

              |- layers.py

              |- losses.py

              |- opt.py

              |- training.py

|- main.py

|- config.py

 

The configuration, folder is contained all the configuration files for each model to train. In each configuration file, there are all the hyper-parameters and parameters concerning the model and the dataset to use for the training. The Dataset folder contains the dataloader to generate the temporal sequences based on the sequence length defined a-priori for the training. The esa folder contains the plot used to investigate the hyper-parameter optimization (HPO) results. Each of these jupyter is associated with a specific architecture, e.g plot_results_conv_ae1D concerning the experiment launched for this architecture. Following the instruction contained in each jupyter it is possible to resume the results of a specific hpo exploration together with the model saved during the experiment. In the folder model there are the scripts that define each model type. The Notebook folder contains the notebook used to resume a model trained from scratch. The model can be trained from scratch using the main.py file once the configuration file name is defined. For example, to train from scratch a conv_ae1D the following command should be run:

Python main.py - -config_name conv_ae1D

The model is trained using the hyper-parameters contained into the configuration file specified with the argument --config_name. The same train can be launched also from the corresponding notebook. In this case, in the notebook/plot_results_conv_ae1D.ipynb it is required to switch the Train flag contained in a cell of the notebook to True. Then training started and its train and validation loss was reported to the jupyter notebook cell. Finally, the configuration file reports all the path needed for the scripts.
