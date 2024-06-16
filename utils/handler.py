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


