import os
import sys

current_directory = os.getcwd()
utils_directory = os.path.abspath(os.path.join(current_directory, '..', '..'))

sys.path.append(utils_directory)

print(os.getcwd())
import utils

import demo