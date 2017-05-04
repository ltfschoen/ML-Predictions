import sys
import logging
from logging import config

class LoggingConfig:
    """ Manually override lowest-severity log message level
    that logger handles from command line by executing with flags:
        i.e. python main.py --log=WARNING

    Sample usage:
        logger.debug('debug message')
        logger.info('info message')
        logger.warning('warn message')
        logger.error('error message')
    """

    def __init__(self):
        self.valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR'] # numeric levels 10, 20, 30, 40
        self.numeric_level = None
        self.setup_logging_level()
        self.numeric_level = logging.getLogger().getEffectiveLevel()

    def setup_logging_level(self):
        """ Load logger config, create logger, and set logging level """
        logging.config.fileConfig('logging.conf')
        logger = logging.getLogger('Predictions Logger')
        logger.setLevel('ERROR') # specifies lowest-severity log message a logger will handle
        if len(sys.argv):
            log_args = [arg for arg in sys.argv if '--log=' in arg]
            if len(log_args) > 0:
                logger.setLevel(self.get_log_level(log_args))

    def get_log_level(self, log_args):
        proposed_level = log_args[0].split("=", 1)[1].upper()
        if not proposed_level in self.valid_levels:
            raise ValueError('Invalid log level: %s' % proposed_level)
        return proposed_level