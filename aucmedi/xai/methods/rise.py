#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from skimage.filters import gaussian
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base


#-----------------------------------------------------#
#                        SHAP                         #
#-----------------------------------------------------#
class Rise(XAImethod_Base):
    """ XAI Method for SHapley Additive exPlanations (SHAP).

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation #1"

    ??? abstract "Reference - Implementation #2"

    ??? abstract "Reference - Publication"

    This class provides functionality for running the compute_heatmap function,

    """
    def __init__(self, model : tf.keras.Model, total_mask : int = 2000, mask_size : int = 8 ,mode : str = None, masks_user = None):
        """ Initialization function for creating a SHAP as XAI Method object.

        Args:
            model (tf.keras.Model):             Model object for which the SHAP method should be applied.
            total_mask (int):                   Number of masks to be generated.
            mask_size (int):                    Size of the mask.
            mode (str):                         Mode of the mask. Choose between 'blur' for Gaussian blur, 'noise' for colored noise, 'noise_bw' for grayscale noise, and None for regular perturbation.
            masks_user (numpy.ndarray):          User-defined masks.

        """
        self.model = model
        self.input_size = self.model.input_shape[1:3]
        self.num_masks = total_mask
        
        self.mode = mode
        self.masks = masks_user if masks_user is not None else self.generate_masks(total_mask, mask_size, 0.5)
        
        self.result = None
    
    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, all_class=False, eps=1e-8):
        image = self.normalize(image)
        if self.result is None:
            self.result, mask = self.masking(image)

        sal_map = self.result[class_index]

        numer = sal_map - np.min(sal_map)
        denom = (sal_map.max() - sal_map.min()) + eps
        sal_map = numer / denom

        return sal_map
            
    
    def masking(self, image):
        """ Core function for computing the SHAP heatmap for a provided image and for specific classification outcome.

        ???+ attention
            Be aware that the image has to be provided in batch format.

        Args:
            image (numpy.ndarray):              Image matrix encoded as NumPy Array (provided as one-element batch).
            class_index (int):                  Classification index for which the heatmap should be computed.
            eps (float):                        Epsilon for rounding.

        The returned heatmap is encoded within a range of [0,1]

        ???+ attention
        """

        verbose = 0
        batch_size = 100

        fudged_image = image.copy()

        if self.mode == 'blur': #Gaussian blur
            fudged_image = gaussian(fudged_image, sigma=4, preserve_range = True)

        elif self.mode == 'noise': #Colored noise
            fudged_image = np.random.normal(255/2,255/9,size = fudged_image.shape).astype('int')

        elif self.mode == 'noise_bw': #Grayscale noise
            fudged_image = np.random.normal(255/2,255/5,size = (fudged_image.shape[:2])).astype('int')
            fudged_image = np.stack((fudged_image,)*3, axis=-1)

        else:
            fudged_image = np.zeros(image.shape) #Regular perturbation with a black gradation


        preds = []

        #Doing these matrix multiplications between the masks and the image can quickly eat up memory.
        #So we multiply the image with one batch of masks at a time and later append the predictions.

        if(verbose):

            print('Using batch size: ',batch_size, flush = True)

        for i in (tqdm(range(0, self.num_masks, batch_size)) if verbose else range(0, self.num_masks, batch_size)):

            masks_batch = self.masks[i:min(i+batch_size, self.num_masks)]
            masked = image*masks_batch + fudged_image*(1-masks_batch)

            to_append = self.model.predict(masked)

            preds.append(to_append)

        preds = np.vstack(preds)

        sal = preds.T.dot(self.masks.reshape(self.num_masks, -1)).reshape(-1, *self.input_size)
        sal = sal / self.num_masks / 0.5

        return sal, self.masks

    #---------------------------------------------#
    #              Generate Masker                #
    #---------------------------------------------#
    def generate_masks(self, N, s, p1):
        """
        Generate a distribution of random binary masks.

        Args:
            N: Number of masks.
            s: Size of mask before upsampling.
            p1: Probability of setting element value to 1 in the initial mask.

        Returns:
            masks: The distribution of upsampled masks.
        """

        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        masks = np.empty((N, *self.input_size))


        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        masks = masks.reshape(-1, *self.input_size, 1)
        return masks
    
    def normalize(self, x):
        """ Normalize the input image.

        Args:
            x (numpy.ndarray):                  Image matrix encoded as NumPy Array.

        The returned image is normalized within a range of [0,1].

        """
        img_float = x.astype(np.float32)

        if img_float.max() > 1:
            x_normalized = (img_float - img_float.min()) / (img_float.max() - img_float.min())
        else:
            x_normalized = img_float
        return x_normalized

    def visualize_heatmap(self, image, heatmap, out_path=None,
                    alpha=0.4, labels=None):
        if image.shape[-1] == 1:
            image = np.concatenate((image,) * 3, axis=-1)

        heatmap = np.uint8(heatmap * 255)
        jet = plt.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap] * 255

        si_img = jet_heatmap * alpha + (1 - alpha) * image
        si_img = si_img.astype(np.uint8)
        pil_img = Image.fromarray(si_img)
        if out_path is None:
            plt.imshow(si_img)
            plt.axis('off')  # Hide the axis
            plt.show()
        else:
            pil_img.save(out_path)
