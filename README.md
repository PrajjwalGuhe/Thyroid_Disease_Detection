# Thyroid Disease Detection

## Overview
This project focuses on the detection of thyroid disease using machine learning techniques. The objective is to build a model that can accurately classify whether a person has a thyroid disease based on various medical attributes.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Dataset
The dataset used in this project contains medical records with several features related to thyroid health. The target variable indicates whether the individual has a thyroid disease or not. The dataset can be found in the `data` directory.

### Features
- `age`: Age of the patient
- `sex`: Gender of the patient
- `on_thyroxine`: Boolean indicating if the patient is on thyroxine medication
- `query_on_thyroxine`: Boolean indicating if the patient is querying on thyroxine
- `on_antithyroid_medication`: Boolean indicating if the patient is on anti-thyroid medication
- `sick`: Boolean indicating if the patient is sick
- `pregnant`: Boolean indicating if the patient is pregnant
- `thyroid_surgery`: Boolean indicating if the patient has undergone thyroid surgery
- `I131_treatment`: Boolean indicating if the patient has undergone I131 treatment
- `query_hypothyroid`: Boolean indicating if the patient is querying for hypothyroidism
- `query_hyperthyroid`: Boolean indicating if the patient is querying for hyperthyroidism
- `TSH_measured`: Boolean indicating if the TSH (Thyroid Stimulating Hormone) was measured
- `TSH`: TSH value
- `T3_measured`: Boolean indicating if the T3 hormone was measured
- `T3`: T3 value
- `TT4_measured`: Boolean indicating if the Total T4 hormone was measured
- `TT4`: Total T4 value
- `T4U_measured`: Boolean indicating if the T4U (Thyroxine Binding Capacity) was measured
- `T4U`: T4U value
- `FTI_measured`: Boolean indicating if the FTI (Free Thyroxine Index) was measured
- `FTI`: FTI value
- `TBG_measured`: Boolean indicating if the TBG (Thyroxine-Binding Globulin) was measured
- `TBG`: TBG value
- `target`: Indicates if the patient has thyroid disease (1) or not (0)

## Installation
To run this project, ensure you have Python installed on your machine. Follow the steps below to set up the environment:

1. Clone the repository
    ```sh
    git clone https://github.com/PrajjwalGuhe/Thyroid_Disease_Detection.git
    cd Thyroid_Disease_Detection
    ```

2. Create a virtual environment
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To train and evaluate the model, run the following command:
```sh
python app.py
```

## Model
The project uses a machine learning model to classify thyroid disease. The model pipeline includes the following steps:

Data Preprocessing
Feature Engineering
Model Training
Model Evaluation
Algorithms
Decision Tree
Random Forest
Support Vector Machine

## Results
The performance of the models is evaluated using accuracy, precision, recall, and F1-score. Detailed results and analysis can be found in the notebooks/exploratory_data_analysis.ipynb.

## Contributing
Contributions are welcome! Please follow these steps to contribute:

Fork the repository
Create a new branch (git checkout -b feature/your-feature-name)
Commit your changes (git commit -m 'Add some feature')
Push to the branch (git push origin feature/your-feature-name)
Create a new Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Prajjwal Guhe - prajjwalguhe@gmail.com.com

Project Link: https://github.com/PrajjwalGuhe/Thyroid_Disease_Detection
