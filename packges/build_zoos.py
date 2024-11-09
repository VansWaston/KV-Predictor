
import os
import logging
from logging.handlers import RotatingFileHandler

def build_docker_image(
    images: list = [],
    summary: str = None,
):
    """_summary_

    Args:
        images (list, optional): list of docker image needed to pull. Defaults to [].
        summary (str, optional): summary of the docker image, set to be the path of the logging file if needed. Defaults to None.
    """
    assert len(images) > 0, "No images to pull"
    
    isTrue = [False] * len(images)
    
    # Set up logging
    if summary is not None:
        if not os.path.exists(os.path.dirname(summary)):
            os.makedirs(os.path.dirname(summary))
        handler = RotatingFileHandler(summary, maxBytes=1000000, backupCount=3)  # 1MB each file, keep 3 files
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                handler,  # output to file
                logging.StreamHandler()  # output to console
            ]
        )
    
    for idx, image in enumerate(images):
        while isTrue[idx] == False:
            try:
                # use proxychains to speed up the process
                tmp = os.popen(f"proxychains docker pull {image}").readlines()
                isTrue[idx] = True
                
                logging.info(f"Pulling {image} successfully")
                logging.info(f"Output: \n{tmp}")
                
                break  # Exit the loop if the command is successful
            except Exception as e:
                print(f"Error: {e}")
                continue