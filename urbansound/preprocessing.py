import os
import random
import warnings
from glob import glob
from typing import Dict, List
from typing import Tuple
from zipfile import ZipFile

import joblib
import librosa
import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm

from utils import ensure_dir


class UrbanSoundData:
    """This class gathers the preprocessed data from the mfcc_dict as well as the
    metadata .csv files and provides easy access to the features plus their
    corresponding labels for train and test data.
    """

    def __init__(self, data_dir: str = os.path.join("..", "data"), n_mfccs: int = 20,
                 n_augmentations: int = 2):
        self.data_dir = data_dir

        mfcc_path = os.path.join(self.data_dir, "mfcc", f"mfcc_{n_mfccs}_aug_{n_augmentations}.z")
        try:
            self.features: Dict[int, List[np.ndarray]] = joblib.load(mfcc_path)
        except FileNotFoundError:
            extractor = UrbanSoundExtractor(data_dir)
            self.features = extractor.prepare_data()

        self.train_short_labels: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, "labels", "train_short.csv"))
        self.train_long_labels: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, "labels", "train_long.csv"))
        self.test_labels: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, "labels", "test.csv"))

    @property
    def train_data_short(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns four numpy arrays. The first two are train and test features, the second
        two are train and test labels. It uses all samples from 'train_short.csv'.

        :returns: train_features, test_features, train_labels, test_labels
        """
        return train_test_split(*self._transform_data(self.train_short_labels))

    @property
    def train_data_long(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns four numpy arrays. The first two are train and test features, the second
        two are train and test labels. It uses all samples from 'train_long.csv'.

        :returns: train_features, test_features, train_labels, test_labels
        """
        features, labels = self._transform_data(self.train_long_labels)
        feature_train, feature_val, label_train, label_val = train_test_split(features, labels)
        # due to image augmentation, every id now has a list of one or more examples instead of
        # just one. All these examples have the same label, but instead of having one dimension
        # for the ID (dimension 0) and one for the label (dimension 1) we only want to have one
        # for label containing all entries of the augmented list. The time and mfcc
        # dimensions (dimensions 2 and 3) are kept.
        # We do this after the train_test_split to not mix augmented images from one sample
        # into both the train and validation sets since they are sometimes very similar and
        # could lead to overfitting.
        n_labels = features.shape[1]
        feature_train = feature_train.reshape(-1, *feature_train.shape[2:])
        feature_val = feature_val.reshape(-1, *feature_val.shape[2:])

        # Due to the augmentation, the labels also need to be repeated as often as we
        # augmented the images
        label_train = np.repeat(label_train.toarray(), n_labels, axis=0)
        label_val = np.repeat(label_val.toarray(), n_labels, axis=0)
        return feature_train, feature_val, label_train, label_val

    @property
    def test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns two numpy arrays. The first one contains the features, the second one
        the corresponding labels. It uses all samples from 'test.csv'.

        :returns: feature, labels
        """
        features, labels = self._transform_data(self.test_labels)
        # due to image augmentation, every id now has a list of one or more examples instead of
        # just one. All these examples have the same label, but instead of having one dimension
        # for the ID (dimension 0) and one for the label (dimension 1) we only want to have one
        # for label containing all entries of the augmented list. The time and mfcc
        # dimensions (dimensions 2 and 3) are kept.
        n_labels = features.shape[1]
        features = features.reshape(-1, *features.shape[2:])

        # Due to the augmentation, the labels also need to be repeated as often as we
        # augmented the images
        labels = np.repeat(labels.toarray(), n_labels, axis=0)
        return features, labels

    def _transform_data(self, label_data_frame: pd.DataFrame) -> Tuple[np.ndarray, csc_matrix]:
        """Picks out features from self.features whose ID is in the label_data_frame and
        converts their class label into one-hot vectors.

        :param label_data_frame: A data frame with an 'ID' and a 'Class' Column
        :return: List of input data and a list of corresponding labels
        """
        label_encoder = LabelEncoder()
        oh_encoder = OneHotEncoder()

        features = [(self.features[str(mfcc_id)]) for mfcc_id in label_data_frame.loc[:, "ID"]]
        features = np.stack(features)

        number_labels = label_encoder.fit_transform(label_data_frame.loc[:, "Class"])
        one_hot_labels = oh_encoder.fit_transform(number_labels.reshape(-1, 1))
        return features, one_hot_labels


class UrbanSoundExtractor:
    """This class handles the download of the data archive as well as
    extracting the archive to the data directory. Furthermore,
    it extracts the MFCC features from the sound files.
    The sounds files will be extracted to *<data_dir>/sounds*, while
    the label files will be extracted to *<data_dir>/labels*, and the
    MFCC features will be extracted to *<data_dir>/mfcc/mfcc_dict.z*
    """

    _DATA_URL = "https://filebox.fhooecloud.at/s/rLADyyMceM77XkU/download"
    _ARCHIVE_NAME = "data.zip"

    def __init__(self, data_dir: str = os.path.join("..", "data")):
        ensure_dir(data_dir)
        self.data_dir = data_dir

    def prepare_data(self, n_mfccs: int = 20, n_augmentations: int = 2) -> dict:
        self.download_data()
        self.extract_archive()
        return self.extract_mfccs(n_mfccs, n_augmentations)

    def download_data(self) -> None:
        """Downloads the data archive from the university filebox and saves
        it in *self.data_dir*
        """
        archive_path = os.path.join(self.data_dir, self._ARCHIVE_NAME)

        if os.path.isfile(archive_path):
            print("Archive already downloaded, skipping")
            return

        if not self._confirm_download():
            print("Skipping download")
            return

        raw_data = requests.get(self._DATA_URL, stream=True)
        raw_data_size = int(raw_data.headers['content-length'])

        try:
            with tqdm(total=raw_data_size, ascii=True,
                      unit_scale=True, desc="Download") as progress_bar:
                with open(os.path.join(self.data_dir, self._ARCHIVE_NAME), 'wb') as fd:
                    chunk_size = 128
                    for chunk in raw_data.iter_content(chunk_size=chunk_size):
                        fd.write(chunk)
                        progress_bar.update(len(chunk))
        except KeyboardInterrupt:
            warnings.warn("User interrupted download, deleting incomplete archive")
            if os.path.isfile(archive_path):
                os.remove(archive_path)
            raise

    @staticmethod
    def _confirm_download() -> bool:
        print("Are you sure you want to download the data archive? (download size is about 1.8GB)")
        answer = ""
        while not answer == "" and not answer.startswith("y") and not answer.startswith("n"):
            answer = input("(y/[n]) >> ").lower()

        return answer.startswith("y")

    def extract_archive(self, force_extract: bool = False) -> None:
        """Extracts the contents of the zip archive and puts it into the
        sounds and labels subdirectories.
        """
        archive_path = os.path.join(self.data_dir, self._ARCHIVE_NAME)
        sound_dir = os.path.join(self.data_dir, "sounds")
        label_dir = os.path.join(self.data_dir, "labels")

        # TODO: Better way to detect already extracted archive
        if not force_extract and os.path.isfile(os.path.join(sound_dir, "0.wav")):
            print("Archive already extracted, skipping")
            return

        try:
            with ZipFile(archive_path) as archive:
                for member_name in tqdm(archive.namelist(), desc="Extraction"):
                    if member_name.endswith(".wav"):
                        archive.extract(member_name, sound_dir)
                    elif member_name.endswith(".csv"):
                        archive.extract(member_name, label_dir)
        except KeyboardInterrupt:
            warnings.warn("User interrupted extraction. Extracted files will not be deleted", RuntimeWarning)
            if os.path.isfile(archive_path):
                os.remove(archive_path)
            raise

    def extract_mfccs(self, n_mfccs: int = 20,
                      n_augmentations: int = 2) -> Dict[int, List[np.ndarray]]:
        """Loads the sound files from the sounds subdirectory and calculates
        the mel-frequency spectrum of each file. Optionally, it will perform
        random pitch and frequency augmentations to the data beforehand.
        The results are saved into a compressed .z file in the mfcc subdirectory.
        """
        sound_dir = os.path.join(self.data_dir, "sounds")
        mfcc_path = os.path.join(self.data_dir, "mfcc", f"mfcc_{n_mfccs}_aug_{n_augmentations}.z")

        if os.path.isfile(mfcc_path):
            print("MFCCs already extracted, skipping")
            return joblib.load(mfcc_path)

        n_cpu = os.cpu_count()
        tuples = Parallel(n_jobs=n_cpu)(
            delayed(_extract_mfcc)(file, sound_dir, n_mfccs, n_augmentations)
            for file in tqdm(glob(os.path.join(sound_dir, "*.wav")), desc="MFCC")
        )

        mfcc_dict = dict(tuples)
        joblib.dump(mfcc_dict, mfcc_path)

        return mfcc_dict


# needs to be a module top-level function to support multi-processing
# don't move this function into a class
def _extract_mfcc(file, sound_dir: str, n_mfccs: int,
                  n_augmentations: int) -> Tuple[int, List[np.ndarray]]:
    max_sound_length = 173
    sound_id = file[len(sound_dir + "/"):-len(".wav")]

    audio, sample_rate = librosa.load(file)

    samples = augment_audio(audio, sample_rate, n_augmentations)

    mfccs = []
    for sample in samples:
        mfcc = librosa.feature.mfcc(sample, sample_rate, n_mfcc=n_mfccs)
        mfcc = librosa.util.fix_length(mfcc, max_sound_length)
        mfccs.append(mfcc)
    return sound_id, mfccs


def audio_change_pitch(audio, sample_rate):
    pitch_change = random.gauss(0, 1)
    pitched_audio = librosa.effects.pitch_shift(audio, sample_rate, pitch_change)
    return pitched_audio


def audio_time_strech(audio, _):
    stretch = random.lognormvariate(0, 0.1)
    stretched_audio = librosa.effects.time_stretch(audio, stretch)
    return stretched_audio


def audio_white_noise(audio, _):
    noise = np.random.normal(0, 0.005)
    return audio + noise


def augment_audio(audio: np.ndarray, sample_rate: int,
                  n_augmentations: int) -> List[Tuple[np.ndarray, int]]:
    augmented_set = [audio]
    augmentations = [audio_change_pitch, audio_time_strech, audio_white_noise]

    for _ in range(n_augmentations):
        augment = random.choice(augmentations)
        augmented_set.append(augment(audio, sample_rate))

    return augmented_set
