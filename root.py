# THIS FILE NEEDS TO BE IN THE ROOT DIRECTORY

import os
from pathlib import Path

from arguments.data_arguments import DataAugmentationType

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = ROOT_DIR / 'data'
AUGMENTED_DIR = DATA_DIR / 'augmented'
TRANSLATION_DIR = AUGMENTED_DIR / DataAugmentationType.TRANSLATION.value
BACK_TRANSLATION_DIR = AUGMENTED_DIR / DataAugmentationType.BACK_TRANSLATION.value