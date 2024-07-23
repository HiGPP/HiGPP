from read_log import ReadLog
from MiDA import MiDA
import sys
import tensorflow as tf

if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_physical_devices('GPU')

    
    tf.config.experimental.set_visible_devices([gpus[1]], 'GPU')
    
    tf.config.experimental.set_memory_growth(gpus[1], True)

    
    
    eventlog = "bpi12_work_all"
    get_log = ReadLog(eventlog).readView()
    MiDA(eventlog).smac_opt()
