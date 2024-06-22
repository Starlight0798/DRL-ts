from loguru import logger
import sys

def raise_warning():
    '''
    Raise an exception instead of a warning.
    Useful for debugging.
    '''
    import warnings
    import logging
    
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
    
    
    
def print2log():
    '''
    Print to loguru logger.
    '''
    import builtins
    _print = builtins.print
    
    def loguru_print(*args, **kwargs):
        message = ' '.join(map(str, args))
        sep = kwargs.get('sep', ' ')
        file = kwargs.get('file', None)
        try:
            if file is None:
                logger.info(message)
            else:
                _print(*args, **kwargs)
        except Exception as e:
            _print(*args, **kwargs)
            
    builtins.print = loguru_print
    logger.debug('print() redirected to loguru logger.')


