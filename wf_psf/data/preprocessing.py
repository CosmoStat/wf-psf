"""Data Preprocessing.

A module to load and preprocess training and test data.

:Authors: Tobias Liaudat <tobiasliaudat@gmail.com> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""


def load_dataset_dict(filename,allow_pickle_flag=True):
    """Load Numpy Dataset Dictionary.

    A function to load dataset dictionary.

    Parameters
    ----------
    filename: str
        Name of file 
    allow_pickle_flag: bool
        Boolean flag to set when loading numpy files
    
    """
    dataset = np.load(filename, allow_pickle=allow_pickle_flag)[()]
    return dataset


def convert_to_tensor(dataset):
    return tf.convert_to_tensor(dataset, dtype=tf.float32)

    #train_SEDs = train_dataset["SEDs"]
    #train_parameters = train_dataset["parameters"]
    #test_dataset = np.load(
    #    args["dataset_folder"] + args["test_dataset_file"], allow_pickle=True
    #)[()]

    #test_SEDs = test_dataset["SEDs"]

    # Convert to tensor
   # tf_noisy_train_stars = tf.convert_to_tensor(
  #      train_dataset["noisy_stars"], dtype=tf.float32)

    #tf_train_pos = tf.convert_to_tensor(train_dataset["positions"], dtype=tf.float32)
    #tf_test_stars = tf.convert_to_tensor(test_dataset["stars"], dtype=tf.float32)
    #tf_test_pos = tf.convert_to_tensor(test_dataset["positions"], dtype=tf.float32)

