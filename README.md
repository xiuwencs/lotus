- Deploy NEMETYL according to https://github.com/vs-uulm/nemesys; 
- Run command "nemesys.py pcapfilename" with pcap in the dataset directory and get the results as text files(fields are separated by commas).
- Run Classifier_protocol module with text files from last step to classify protocols.
- If dataset is a binary protocol, run LinearRegression module.
- If dataset is a textual protocol, run Word_segmentation module first and then run LinearRegression module.