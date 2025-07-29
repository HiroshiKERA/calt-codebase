import os
import re

def get_checkpoint_id(save_dir):
    """
    Get the checkpoint id from the save directory.
    """
    cpt_file = [f for f in os.listdir(save_dir) if 'checkpoint' in f][0]
    cpid = int(re.search(r'checkpoint-(\d+)', cpt_file).group(1))
    return cpid 