# MathSymbolSegmentor

## Configurations

- ### Split

    `python symbolsegmentor.py split [path to data] [output file name]`

    `ex. python symbolsegmentor.py split ..\inkml\ splitName`

	....
*creates two files with splitName_train_data.txt and splitName_test_data.txt.
Each file contains inkml expression file names.*
	....

- ### Create

    `python symbolsegmentor.py create [train_file name] [junk_file name] [test file name] [junk test file name] [path to data] [path to junk]`

    `ex. python symbolsegmentor.py create splitName_train_data.txt junk_train.txt splitName_test_data.txt junk_test.txt ..\inkml\ ..\junk\`

	....
*creates training and testing feature vectors for digits and junk and saves in the system.
It also saves the labels, class dictionary maps, ground truth.*
	....

- ### Train

    `python symbolsegmentor.py train [0/1]`

    `ex. python symbolsegmentor.py train 0/1`

	....
*Config 0 is Train with junk data while Config 1 is train without junk.*
*Reads the feature vectors and labels, classes and ground truths and trains the classifier model.
Then fine tuning is done where all models are saved in a separate folder along with results txt file.*
	....

- ### Evaluate

    `python symbolsegmentor.py evaluate [segmenter_name] [files] [path to data] [path to output]`

    `ex. python symbolsegmentor.py evaluate baseline/kmean splitName_train_data.txt ..\inkml\ ..\output\`

	....
*Evaluates the expression from the file and using the segmenter mentioned, creates .lg file with output in output path.*
	....