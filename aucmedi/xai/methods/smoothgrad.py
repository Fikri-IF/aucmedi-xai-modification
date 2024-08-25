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
# External Libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

from PIL import Image
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#           Saliency Maps / Backpropagation           #
#-----------------------------------------------------#
class SmoothGrad(XAImethod_Base):
    """ XAI Method for Saliency Map (also called Backpropagation).

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation #1"
        Author: Yasuhiro Kubota <br>
        GitHub Profile: [https://github.com/keisen](https://github.com/keisen) <br>
        Date: Aug 11, 2020 <br>
        [https://github.com/keisen/tf-keras-vis/](https://github.com/keisen/tf-keras-vis/) <br>

    ??? abstract "Reference - Implementation #2"
        Author: Huynh Ngoc Anh <br>
        GitHub Profile: [https://github.com/experiencor](https://github.com/experiencor) <br>
        Date: Jun 23, 2017 <br>
        [https://github.com/experiencor/deep-viz-keras/](https://github.com/experiencor/deep-viz-keras/) <br>

    ??? abstract "Reference - Publication"
        Karen Simonyan, Andrea Vedaldi, Andrew Zisserman. 20 Dec 2013.
        Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.
        <br>
        [https://arxiv.org/abs/1312.6034](https://arxiv.org/abs/1312.6034)

    This class provides functionality for running the compute_heatmap function,
    which computes a Saliency Map for an image with a model.
    """
    def __init__(self, model : tf.keras.Model, smooth_samples=None, smooth_noise=None):
        """ Initialization function for creating a Saliency Map as XAI Method object.

        Args:
            model (keras.model):               Keras model object.
            smooth_samples (int):              Number of samples for SmoothGrad computation.
            smooth_noise (float):              Noise level for SmoothGrad computation.
        """
        # Cache class parameters
        self.model = model
        self.smooth_samples = smooth_samples
        if self.smooth_samples is None : self.smooth_samples = 20

        self.smooth_noise = smooth_noise
        if self.smooth_noise is None : self.smooth_noise = 0.20

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, all_class=False, eps=1e-8):
        """ Core function for computing the Saliency Map for a provided image and for specific classification outcome.

        ???+ attention
            Be aware that the image has to be provided in batch format.

        Args:

        The returned heatmap is encoded within a range of [0,1]

        ???+ attention
            The shape of the returned heatmap is 2D -> batch and channel axis will be removed.

        Returns:
            heatmap (numpy.ndarray):            Computed Saliency Map for provided image.
        """
        # Compute gradient for desierd class index
        print(image.shape)
        score= CategoricalScore([class_index])

        replace2linear = self.model_modifier_function()
        saliency = Saliency(self.model, model_modifier=replace2linear,clone=True)
        saliency_map = saliency(score,
                        image,
                        self.smooth_samples, # The number of calculating gradients iterations.
                        self.smooth_noise) # noise spread level.
        saliency_map = normalize(saliency_map)
        print(saliency_map.shape)
        print("sliced salienc map:",np.squeeze(saliency_map))
        return np.squeeze(saliency_map)



    def visualize_heatmap(self, image, heatmap, out_path=None,
                    alpha=0.4, labels=None):
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))  # Adjusted figsize for a single image

        ax.set_title(labels[0][0], fontsize=16)

        ax.imshow(heatmap, cmap='jet')


        ax.axis('off')

        # Adjust the layout and display the image
        plt.tight_layout()
        plt.show()
        # if image.shape[-1] == 1:
        #     image = np.concatenate((image,) * 3, axis=-1)
        # print(labels.shape)
        # print(heatmap.shape)

        # heatmap = np.uint8(heatmap * 255)
        # jet = plt.get_cmap("jet")
        # jet_colors = jet(np.arange(256))[:, :3]
        # jet_heatmap = jet_colors[heatmap] * 255

        # si_img = jet_heatmap * alpha + (1 - alpha) * image
        # si_img = si_img.astype(np.uint8)
        # pil_img = Image.fromarray(si_img)
        # if out_path is None:
        #     plt.imshow(si_img)
        #     plt.axis('off')  # Hide the axis
        #     plt.show()
        # else:
        #     pil_img.save(out_path)

    def model_modifier_function(self):
        self.model.layers[-1].activation = tf.keras.activations.linear