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
from lime import lime_image
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#                  LIME: Con Features                 #
#-----------------------------------------------------#
class LimeCon(XAImethod_Base):
    """ XAI Method for LIME Con.

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation"
        Lime: Explaining the predictions of any machine learning classifier <br>
        GitHub: [https://github.com/marcotcr/lime](https://github.com/marcotcr/lime) <br>

    ??? abstract "Reference - Publication"
        Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. 9 Aug 2016.
        "Why Should I Trust You?": Explaining the Predictions of Any Classifier
        <br>
        [https://arxiv.org/abs/1602.04938](https://arxiv.org/abs/1602.04938)

    This class provides functionality for running the compute_heatmap function,
    which computes a Lime Con Map for an image with a model.
    """
    def __init__(self, model : tf.keras.Model, num_eval=None):
        """ Initialization function for creating a Lime Con Map as XAI Method object.

        Args:
            model (keras.model):            Keras model object.
            layerName (str):                Not required in LIME Pro/Con Maps, but defined by Abstract Base Class.
            num_eval (int):                 Number of samples for LIME instance explanation.
        """
        # Cache class parameters
        self.model = model
        self.num_samples = num_eval
        if self.num_samples is None : self.num_samples = 1000

    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, all_class=False, eps=1e-8):
        """ Core function for computing the Lime Con Map for a provided image and for specific classification outcome.

        ???+ attention
            Be aware that the image has to be provided in batch format.

        Args:
            image (numpy.ndarray):              Image matrix encoded as NumPy Array (provided as one-element batch).
            class_index (int):                  Classification index for which the heatmap should be computed.
            eps (float):                        Epsilon for rounding.

        The returned heatmap is encoded within a range of [0,1]

        ???+ attention
            The shape of the returned heatmap is 2D -> batch and channel axis will be removed.

        Returns:
            heatmap (numpy.ndarray):            Computed Lime Con Map for provided image.
        """
        # Initialize LIME explainer
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(image[0].astype("double"),
                                self.model.predict, hide_color=0,
                                labels=(class_index,),
                                num_samples=self.num_samples)
        # Obtain CON explanation mask
        temp, mask = explanation.get_image_and_mask(class_index, hide_rest=True,
                                positive_only=False, negative_only=True)
        heatmap = mask
        # Intensity normalization to [0,1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        # Return the resulting heatmap
        return heatmap

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