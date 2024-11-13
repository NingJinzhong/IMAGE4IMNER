# Breaking the Boundaries: A Unified Framework for Chinese Named Entity Recognition Across Text and Speech

## Task NAME
Integrated Modal Named Entity Recognition（IMNER）

## How to Run the Code

To get the code up and running, please follow these steps:

1. **Install Dependencies**: Start by installing the required packages listed in `requirements.txt`. You can do this by running the following command in your terminal:
   ```
   source create_env.sh
   ```

2. **Download Pre-trained Model**: Next, download the MMSpeech pre-trained model by executing the script `downloadmodel.py`. Run the following command:
   ```
   python downloadmodel.py
   ```

3. **Prepare Dataset**: Obtain the AISHELL dataset from [OpenSLR](https://www.openslr.org/33/). After downloading, ensure to set the path to this dataset in the `train.py` file by assigning it to the `audiodatadir` variable.

4. **Configure and Train**: Modify the hyperparameters for IMAGE configuration in `train.py` as per your requirements. Once configured, initiate the training process by executing:
   ```
   python train.py
   ```
