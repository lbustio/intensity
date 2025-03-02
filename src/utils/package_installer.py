import subprocess
from src.utils.constants import PACKAGES
import logging

class PackageInstaller:
    """
    A class to handle the installation of required packages for the project.
    This class uses subprocess to install packages and logs the process using an external logger.
    """

    def __init__(self, install_flag=True, logger=None):
        """
        Initialize the PackageInstaller class.

        Parameters:
            install_flag (bool): Flag to control whether packages should be installed.
            logger (logging.Logger): Logger instance to use for logging.
        """
        self.install_flag = install_flag
        self.logger = logger  # Use the provided logger, or a default one if None
        if not self.logger:
            raise ValueError("Logger must be provided if no default logger is set.")

    def install_packages(self):
        """
        Install the required packages by using pip subprocess calls.

        If the `install_flag` is set to False, the installation process is skipped.
        """
        if self.install_flag:
            self.logger.info("📦 Installing required libraries...")

            # Install each package using pip in a subprocess
            for package in PACKAGES:
                self.logger.info(f"Installing '{package}'...")
                try:
                    # Run pip install command with error checking
                    subprocess.run(["pip", "install", package],
                                    check=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
                    self.logger.info(f" '{package}' package installed correctly!")
                except subprocess.CalledProcessError as e:
                    # Log any installation errors with the error message
                    self.logger.error(f"Error installing '{package}': {e.stderr}")
                    
            # Generate a 'requirements.txt' file containing all installed Python packages.
            try:
                self.logger.info("Generating 'requirements.txt' file with installed packages...")
                subprocess.run(["pip", "freeze"], check=True, stdout=open("requirements.txt", "w"))
                self.logger.info(" 'requirements.txt' file has been generated successfully!")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error generating 'requirements.txt': {e.stderr}")

            # Inform user that kernel restart is needed
            self.logger.info("🔄 Installation complete. Please restart the kernel manually from VS Code.")
        else:
            # Message when installation is skipped
            self.logger.warning("🚫 Library installation is disabled.")