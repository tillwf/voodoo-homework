import os
import logging
import coloredlogs
import absl.logging

# Remove TensorFlow 2 Info/ Warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
absl.logging.set_verbosity(absl.logging.ERROR)

# Define our logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
coloredlogs.install(level='INFO')
