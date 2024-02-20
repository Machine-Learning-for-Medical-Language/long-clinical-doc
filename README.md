# LCD benckmark: Long clinical document benchmark on mortality prediction


Paper under review. Citation information and paper informaion will follow.

### Platforms

[Link to LCD benchmark CodaBench](TBA)

* LCD benchmark suports CodaBench, which is an online dataset evaluation platform. Please submit your model's output through CodaBench and compare it with others. 

Future plans:
* Full preprocessing code will available soon.
* We are planning to expand our data distribution platforms
* Demo codes are available on [CNLPT library](https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers) as stated in the paper. We will soon release tutorial documents for replicating the results. 

## License:
This repository only provides hashed (i.e. one-way encryption) document ids and the labels of each datapoints. You will need to download notes from PhysioNet.org with fully executed MIMIC-VI data use agreement. Agreeing to [PhysioNet Credentialed Health Data Use Agreement 1.5.0](https://physionet.org/content/mimiciv/view-dua/2.2/) and [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimiciv/view-license/2.2/) is required to utilize the benchmark dataset. 

**Please get access to both [MIMIC-VI v2.2](https://physionet.org/content/mimiciv/2.2/) and [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/).**


<hr>

## How to prepare the dataset

#### Environments: 
We strongly recommand to use Python version over 3.9.
<br>The example codes are tested on Ubuntu 22.04 (python v3.10.12) and MacOS v13.5.2 (python v3.9.6).
<br>pandas and tqdm are required. To install these: `pip install -r requirements.txt`

Please pay attention to the message (stdout) at the end of processing run, as it will tell the integrity of the created data.
Note that the integrety do not check the order of the instances in datasets.

### Folder structure

```bash
create_data.py : Code that merges labels.json with MIMIC-VI note data. 
create_data_serverside.py
```
Please note that `create_data_serverside.py` is the code the authors used to extract labels from pre-processed data. It is not required for users to run the code.


### Steps: 

1. Complete the steps required by [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/) and download `mimic-iv-note-deidentified-free-text-clinical-notes-2.2.zip` file.

2. Unzip the downloaded file. Following is an example bash script for linux users :
```bash
#cd <MOVE_TO_DOWNLOAD_FOLDER>
unzip mimic-iv-note-deidentified-free-text-clinical-notes-2.2.zip
cd note/
gzip -d discharge.csv.gz
export NOTE_PATH=${PWD}/discharge.csv
```
The path to discharge.csv (3.3G) is stored in `$NOTE_PATH`
```bash
shasum -a 1 discharge.csv # sha1 value
# -> c4f0cfcd00bb8cbb118b1613a5c93f31a361e82b  discharge.csv
# Or a9ac402818385f6ab5a574b4516abffde95d641a  discharge.csv (old version)
```

3. Clone this repository 
```bash
# Move to any project folder
# e.g.) cd <MOVE_TO_ANY_PROJECT_FOLDER>
git clone https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc.git
cd long-clinical-doc
```

4. Merge notes and the labels
```bash
export TASK_NAME="out_hos_30days_mortality"
export LABEL_PATH=${TASK_NAME}/labels.json
export OUTPUT_PATH=${TASK_NAME} 

python create_data.py \
 --label_path ${LABEL_PATH} \
 --discharge_path ${NOTE_PATH} \
 --output_path ${OUTPUT_PATH}
```

5. Please make it sure to check the number of processed datapoints. 
If the numbers match the values written below, then the contents of the datasets are identical to our version. 
(Note that this integrity check does not verify the order of instances in the datasets.)
```
Train: 34,759 
Dev: 7,505
Test: 7,568 
```
