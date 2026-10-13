class TTSUnavailableError(RuntimeError):
    """Text-to-speech couldn't be produced (no key, provider down, quota) -
    the route turns it into a 503."""
