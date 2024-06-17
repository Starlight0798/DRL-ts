from loguru import logger
import logging
import sys

def raise_warning():
    '''
    Raise an exception instead of a warning.
    Useful for debugging.
    '''
    import warnings
    
    def warning_handler(message, category, filename, lineno, file, line=None):
        with logger.catch():
            raise Exception(f"{filename}:{lineno}: {category.__name__}: {message}")
        logger.error('[DEBUG] Process killed because of warning!')
        sys.exit(0)
        
    def logger_handler(*args):
        with logger.catch():
            raise Exception(*args)
        logger.error('[DEBUG] Process killed because of Logging warning!')
        sys.exit(0)
        
    warnings.showwarning = warning_handler
    logging.Logger.warning = logger_handler
    
    
    
def print(*args, **kwargs):
    '''
    Print to loguru logger.
    '''
    sep = kwargs.pop('sep', ' ')
    end = kwargs.pop('end', '\n')
    logger.info(sep.join(map(str, args)) + end)


