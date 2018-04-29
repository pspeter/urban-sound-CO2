import requests
from zipfile import ZipFile
import os
import logging
from glob import glob
import joblib
from typing import Tuple, List

from tqdm import tqdm
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from utils import ensure_dir, cached_property

logger = logging.getLogger()
logger.setLevel("DEBUG")


class UrbanSoundData:
    def __init__(self, data_dir: str=os.path.join("..", "data")):
        self.data_dir = data_dir

        mfcc_path = os.path.join(self.data_dir, "mfcc", "mfcc_dict.z")
        try:
            self.features = joblib.load(mfcc_path)
        except FileNotFoundError:
            logging.warning("Data not loaded yet. Use UrbanSoundExtractor to load data first")
            logging.warning("Creating UrbanSoundExtractor to prepare data")
            extractor = UrbanSoundExtractor(data_dir)
            self.features = extractor.prepare_data()

        self.train_short_labels: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, "labels", "train_short.csv"))
        self.train_long_labels: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, "labels", "train_long.csv"))
        self.test_labels: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, "labels", "test.csv"))

    @cached_property
    def train_data_short(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_data(self.train_short_labels)

    @cached_property
    def train_data_long(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_data(self.train_long_labels)

    @cached_property
    def test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_data(self.test_labels)

    def _get_data(self, label_data_frame) -> Tuple[np.ndarray, np.ndarray]:
        label_encoder = LabelEncoder()
        oh_encoder = OneHotEncoder()

        features = [self.features[mfcc_id] for mfcc_id in label_data_frame.loc[:, "ID"]]
        features = np.stack(features)
        number_labels = label_encoder.fit_transform(label_data_frame.loc[:, "Class"])
        one_hot_labels = oh_encoder.fit_transform(number_labels.reshape(-1, 1))
        return features, np.array(one_hot_labels)


class UrbanSoundExtractor:
    """This class handles the download of the data archive as well as
    extracting the archive to the data directory. Furthermore,
    all sounds files will be extracted to *<data_dir>/sounds*, while
    the label files will be extracted to *<data_dir>/labels*.
    """

    _DATA_URL = "https://filebox.fhooecloud.at/s/rLADyyMceM77XkU/download"
    _ARCHIVE_NAME = "data.zip"

    def __init__(self, data_dir: str=os.path.join("..", "data")):
        ensure_dir(data_dir)
        self.data_dir = data_dir

    def prepare_data(self) -> dict:
        self.download_data()
        self.extract_archive()
        return self.extract_mfccs()

    def download_data(self) -> None:
        logging.debug(f"Downloading {self._ARCHIVE_NAME} from {self._DATA_URL}")

        archive_path = os.path.join(self.data_dir, self._ARCHIVE_NAME)

        if os.path.isfile(archive_path):
            logging.info("Archive already downloaded, skipping")
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
            logging.warning("User interrupted download, deleting incomplete archive")
            if os.path.isfile(archive_path):
                os.remove(archive_path)
            raise

    def extract_archive(self, force_extract: bool=False) -> None:
        logging.debug(f"Extracting {self._ARCHIVE_NAME}")

        archive_path = os.path.join(self.data_dir, self._ARCHIVE_NAME)
        sound_dir = os.path.join(self.data_dir, "sounds")
        label_dir = os.path.join(self.data_dir, "labels")

        # TODO: Better way to detect already extracted archive
        if not force_extract and os.path.isfile(os.path.join(sound_dir, "0.wav")):
            logging.info("Archive already extracted, skipping")
            return

        try:
            with ZipFile(archive_path) as archive:
                for member_name in tqdm(archive.namelist(), desc="Extraction"):
                    if member_name.endswith(".wav"):
                        archive.extract(member_name, sound_dir)
                    elif member_name.endswith(".csv"):
                        archive.extract(member_name, label_dir)
        except KeyboardInterrupt:
            logging.warning("User interrupted extraction. Extracted files will not be deleted")
            if os.path.isfile(archive_path):
                os.remove(archive_path)
            raise

    def extract_mfccs(self) -> dict:
        logging.debug(f"Extracting MFCC features")

        sound_dir = os.path.join(self.data_dir, "sounds")
        mfcc_path = os.path.join(self.data_dir, "mfcc", "mfcc_dict.z")

        if os.path.isfile(mfcc_path):
            logging.info("MFCCs already extracted, skipping")
            return joblib.load(mfcc_path)

        mfcc_dict = {}
        max_sound_length = 173

        for file in tqdm(glob(os.path.join(sound_dir, "*.wav")), desc="MFCC"):
            sound_id = file[len(sound_dir + "/"):-len(".wav")]
            mfcc = librosa.feature.mfcc(*librosa.load(file))
            mfcc = librosa.util.fix_length(mfcc, max_sound_length)
            mfcc_dict[int(sound_id)] = mfcc

        joblib.dump(mfcc_dict, mfcc_path)

        return mfcc_dict
