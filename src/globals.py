# Define different run configurations.
CONFIG_normal = {'CONFIG': 'config_normal',
                 'OUTPUT_DIR': 'output',
                 'PICKLED_DIR': 'pickled',
                 'DATA_DIR': 'data'}

CONFIG_test = {'CONFIG': 'config_test',
               'OUTPUT_DIR': 'testoutput',
               'PICKLED_DIR': 'testpickled',
               'DATA_DIR': 'data'}

# Select a configuration to run.
CONFIG = CONFIG_normal
# In order to run 'normal', just uncomment the following line, do NOT delete it
# CONFIG = CONFIG_test
