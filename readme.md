An ML exersise: UCI Machine Learning Repository: Wine Quality Dataset
Tools used: pandas, scikit-learn, pytorch

Cleaning/preproccsing data with pandas.
clean.py; 
    loads the dataset from where it's saved on the device.
    explores the data by displaying statistical summaries, data types, null values.
    handles missing values by filling with the mean of it's column.
    uses IQR to identify and remove outliers in the data.
    uses Z-score normalization to ensure numerical features have the same scale.
    saves the cleaned data for further use.

PyTorch DataLoader to create an efficient data loading pipeling for training.
loader.py;
    uses pytorch DataLoader to to load the processed data.
    split method from scikit-learn to divid the data into trainin/testing sets.
    converts the data to pytorch dataset.
    initiliazes the data loader. (return train_loader swappable with test_loader)

Analyzing prepared data & executing ML tasks
model.py;
    torch.nn to design a small feedforward neural network with one hidden layer.

train.py;
    runs WineQualityClassifier based on user input metrics.
    returns loss w/ epoch count to verify learning & optimization.

[task list if moving on]:
Evaluate a model's accuracy on Validation and/or Test Set



