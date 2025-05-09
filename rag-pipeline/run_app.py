import os
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the Streamlit application."""
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the path to the app
        app_path = os.path.join(script_dir, "src", "web", "app.py")
        
        # Verify the app file exists
        if not os.path.exists(app_path):
            raise FileNotFoundError(f"App file not found at: {app_path}")
        
        # Run the Streamlit app
        logger.info(f"Starting Streamlit app from: {app_path}")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
    except FileNotFoundError as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Streamlit app: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 