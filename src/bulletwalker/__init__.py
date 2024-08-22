from . import logging
import importlib.resources

PACKAGE_NAME = "bulletwalker"

logging.configure(logging.LoggingLevel.INFO)

ASSETS_PATH = importlib.resources.files(__package__) / "assets"
