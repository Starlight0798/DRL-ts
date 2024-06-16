from loguru import logger


def raise_warning():
    '''
    Raise an exception instead of a warning.
    Useful for debugging.
    '''
    import warnings
    
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        with logger.catch(reraise=False):
            raise Exception(f"{filename}:{lineno}: {category.__name__}: {message}")
        logger.error('[DEBUG] Process killed because of warning!')
        exit(0)
        
    warnings.showwarning = warning_handler
    
    
    
def print(*args, **kwargs):
    '''
    Print to loguru logger.
    '''
    sep = kwargs.pop('sep', ' ')
    end = kwargs.pop('end', '\n')
    logger.info(sep.join(map(str, args)) + end)


