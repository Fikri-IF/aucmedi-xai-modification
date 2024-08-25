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
import shap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Internal Libraries
from aucmedi.xai.methods.xai_base import XAImethod_Base

#-----------------------------------------------------#
#                        SHAP                         #
#-----------------------------------------------------#
class Shap(XAImethod_Base):
    """ XAI Method for SHapley Additive exPlanations (SHAP).

    Normally, this class is used internally in the [aucmedi.xai.decoder.xai_decoder][] in the AUCMEDI XAI module.

    ??? abstract "Reference - Implementation #1"
        Author: Lundberg, Scott M and Lee, Su-In <br>
        Date: August 23, 2024 <br>
        [https://github.com2/shap/shap?tab=readme-ov-file](https://github.com/shap/shap?tab=readme-ov-file) <br>

    ??? abstract "Reference - Publication"
        Lundberg, Scott M and Lee, Su-In. 2017.
        A Unified Approach to Interpreting Model Predictions.
        <br>
        [http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)

    This class provides functionality for running the compute_heatmap function,

    """
    def __init__(self, model : tf.keras.Model, num_eval=None):
        """ Initialization function for creating a SHAP as XAI Method object.

        Args:
            model (keras.model):               Keras model object.
            layerName (str):                   Layer name of the convolutional layer for heatmap computation.
            num_eval (int):                    Number of evaluations for SHAP computation.
        """
        # Cache class parameters
        self.model = model
        self.max_eval = num_eval
        if self.max_eval is None : self.max_eval = 500

        # define a masker that is used to mask out partitions of the input image.
        self.masker = self.define_masker()
        self.explainer = shap.Explainer(self.model, self.masker)
    
    #---------------------------------------------#
    #             Heatmap Computation             #
    #---------------------------------------------#
    def compute_heatmap(self, image, class_index, all_class=False, eps=1e-8):
        """ Core function for computing the XAI heatmap for a provided image and for specific classification outcome.

        ???+ attention
            Be aware that the image has to be provided in batch format.

        Args:
            image (numpy.ndarray):              Image matrix encoded as NumPy Array (provided as one-element batch).
            class_index (int):                  Classification index for which the heatmap should be computed.
            all_class (bool):                   If True, the heatmap should be computed for all classes.
            eps (float):                        Epsilon for rounding.

        The returned heatmap should be encoded within a range of [0,1]

        ???+ attention
            The shape of the returned heatmap is 2D -> batch and channel axis will be removed.

        Returns:
            heatmap (numpy.ndarray):            Computed XAI heatmap for provided image.
        """
        img_float = image.astype(np.float32)

        if img_float.max() > 1:
            img_normalized = (img_float - img_float.min()) / (img_float.max() - img_float.min())
        else:
            img_normalized = img_float
        # Compute SHAP values
        shap_values = self.explainer(img_normalized, max_evals=self.max_eval,batch_size=50,
                                     outputs = [class_index])
        shap_exp = shap_values
        if len(shap_exp.output_dims) == 1:
            shap_values = [shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])]
        elif len(shap_exp.output_dims) == 0:
            shap_values = shap_exp.values
        else:
            raise Exception("Number of outputs needs to have support added!! (probably a simple fix)")

        shap_values = np.array(shap_values)
        shap_values = shap_values.squeeze()

        return shap_values
    
    #---------------------------------------------#
    #              Define Masker                  #
    #---------------------------------------------#
    def define_masker(self):
        """ Internal function. Define a masker for SHAP computation.

        The masker is used to mask out partitions of the input image.
        """

        if len(self.model.input_shape) == 4:
            blur_shape = 'blur(' + str(self.model.input_shape[1]) + ',' + str(self.model.input_shape[2]) + ')'
            masker = shap.maskers.Image(blur_shape, self.model.input_shape[1:4])
            return masker
        raise ValueError("SHAP masker not defined for input shape: {}".format(self.model.input_shape))

    def visualize_heatmap(self, image, heatmap, out_path=None,
                    alpha=0.4, labels=None):
        
        width = 20
        aspect = 0.2
        hspace = 0.2

        pixel_values = image.astype(np.float32)
        pixel_values = (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min())

        if not isinstance(heatmap, list):
            heatmap = [heatmap]

        if len(heatmap[0].shape) == 3:
            heatmap = [v.reshape(1, *v.shape) for v in heatmap]
            pixel_values = pixel_values.reshape(1, *pixel_values.shape)

        if labels is not None:
            if isinstance(labels, list):
                labels = np.array(labels).reshape(1, -1)

        x = pixel_values
        fig_size = np.array([3 * (len(heatmap) + 1), 1.5 * (x.shape[0] + 1)])
        if fig_size[0] > width:
            fig_size *= width / fig_size[0]
        fig, axes = plt.subplots(nrows=x.shape[0], ncols=len(heatmap) + 1, figsize=fig_size)
        if len(axes.shape) == 1:
            axes = axes.reshape(1, axes.size)
        for row in range(x.shape[0]):
            x_curr = x[row].copy()

            if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
                x_curr = x_curr.reshape(x_curr.shape[:2])

            if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
                x_curr_gray = 0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2]
                x_curr_disp = x_curr
            else:
                x_curr_gray = x_curr
                x_curr_disp = x_curr

            axes[row, 0].imshow(x_curr_disp, cmap=plt.get_cmap('gray'))
            axes[row, 0].axis('off')

            if len(heatmap[0][row].shape) == 2:
                abs_vals = np.stack([np.abs(heatmap[i]) for i in range(len(heatmap))], 0).flatten()
            else:
                abs_vals = np.stack([np.abs(heatmap[i].sum(-1)) for i in range(len(heatmap))], 0).flatten()
            max_val = np.nanpercentile(abs_vals, 99.9)

            for i in range(len(heatmap)):
                if labels is not None:
                    axes[row, i + 1].set_title(labels[row, i])
                sv = heatmap[i][row] if len(heatmap[i][row].shape) == 2 else heatmap[i][row].sum(-1)
                axes[row, i + 1].imshow(x_curr_gray, cmap=plt.get_cmap('gray'), alpha=alpha, extent=(-1, sv.shape[1], sv.shape[0], -1))
                im = axes[row, i + 1].imshow(sv, cmap=shap.plots.colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
                axes[row, i + 1].axis('off')

        fig.subplots_adjust(hspace=hspace)
        cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0] / aspect)
        cb.outline.set_visible(False)

        if out_path is None:
            plt.show()
        else:
            fig.savefig(out_path)