import matplotlib.pyplot as plt
import os
from typing import List, Optional
from imgflow.workflow import Workflow
from imgflow.transform import *
from tqdm import tqdm

def file_names(path: str) -> List[str]:
    """
    List all files recursively in directory in the specified path.

    Args:
        path: path to directory

    Returns: list of file paths

    """
    all_files = []
    for root, _, files in tqdm(os.walk(path)):
        for filename in files:
            all_files.append(os.path.join(root, filename))
    return all_files


def plot(*args):
    n_subplot = len(args)
    fig, ax = plt.subplots(1, n_subplot)
    for i in range(n_subplot):
        ax[i].imshow(cv2.cvtColor(args[i], cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    all_files = file_names('data')
    for f in all_files:
        myflow1 = Workflow()
        myflow1.add_transform_step(TransformToBW)
        myflow1.add_transform_step(TransformRemoveBWBackgound)
        myflow1.source.from_jpg(f)
        X1 = myflow1.run()


        myflow2 = Workflow()
        myflow2.add_transform_step(TransformToBW)
        myflow2.add_transform_step(TransformRemoveBackgroundAvg)
        myflow2.add_transform_step(TransformBoxCut)
        myflow2.source.from_jpg(f)
        X2 = myflow2.run()


        myflow3 = Workflow()
        myflow3.add_transform_step(TransformRemoveBackgroundAvg)
        myflow3.add_transform_step(TransformBoxCut)
        myflow3.add_transform_step(TransformSquare)
        myflow3.add_transform_step(TransformResize, new_size=400)
        myflow3.source.from_jpg(f)
        X3 = myflow3.run()

        plot(myflow1.source.orig_img(), X1[0], X2[0], X3[0])
