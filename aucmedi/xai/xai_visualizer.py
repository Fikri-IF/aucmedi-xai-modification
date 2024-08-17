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
import os
import shap
import matplotlib.pyplot as plt


from PIL import Image
# from aucmedi.xai import XAIResult

class XAIResult():
    def __init__(self, xai_maps : np.ndarray, images: np.ndarray, labels, samples):
        self.xai_maps = xai_maps
        self.images = images
        self.labels = labels
        self.samples = samples

    def __visualize(self, image, heatmap, out_path=None,
                    alpha=0.4, labels=None, width=20,
                    aspect=0.2, hspace=0.2,
                    cmap=shap.plots.colors.red_transparent_blue, invert=False):
        if invert: image = -image
        if len(heatmap.shape) == 3:
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
                    im = axes[row, i + 1].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
                    axes[row, i + 1].axis('off')

            if hspace == 'auto':
                fig.tight_layout()
            else:
                fig.subplots_adjust(hspace=hspace)
            cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0] / aspect)
            cb.outline.set_visible(False)

            if out_path is None:
                plt.show()
            else:
                fig.savefig(out_path)

        else:
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

    def visualize_xai(self, alpha=0.4, invert=False, out_path=None):
        """
        Visualizes the XAI results.

        Parameters:
        - xai_result: XAIResult object containing images, xai_maps, labels, and samples.
        - alpha: float, opacity level for the XAI map overlay.
        - invert: bool, whether to invert the colors.
        - out_path: str, directory to save the visualized images.
        """

        # images = xai_result.get_images()
        # maps = xai_result.get_xai_maps()
        # labels = xai_result.get_labels()
        # samples = xai_result.get_samples()

        num_classes = self.xai_maps.shape[1]

        for i in range(len(self.images)):
            for j in range(num_classes):
                file_name = None
                if out_path is not None:

                    os.makedirs(out_path, exist_ok=True)

                    file_name = f"{self.labels[i][j]}_{self.samples[i]}"
                    if os.sep in file_name : file_name = file_name.replace(os.sep, ".")
                    file_name = os.path.join(out_path, file_name)
                    
                self.__visualize(self.images[i], self.xai_maps[i][j], out_path=file_name, alpha=alpha, labels=self.labels, invert=invert)

# def visualize_xai(xai_result : XAIResult, alpha=0.4, invert=False, out_path=None):
#     """
#     Visualizes the XAI results.

#     Parameters:
#     - xai_result: XAIResult object containing images, xai_maps, labels, and samples.
#     - alpha: float, opacity level for the XAI map overlay.
#     - invert: bool, whether to invert the colors.
#     - out_path: str, directory to save the visualized images.
#     """

#     images = xai_result.get_images()
#     maps = xai_result.get_xai_maps()
#     labels = xai_result.get_labels()
#     samples = xai_result.get_samples()

#     num_classes = maps.shape[1]

#     for i in range(len(images)):
#         for j in range(num_classes):
#             file_name = None
#             if out_path is not None:

#                 os.makedirs(out_path, exist_ok=True)

#                 file_name = f"{labels[i][j]}_{samples[i]}"
#                 if os.sep in file_name : file_name = file_name.replace(os.sep, ".")
#                 file_name = os.path.join(out_path, file_name)
                
#             visualize(images[i], maps[i][j], out_path=file_name, alpha=alpha, labels=labels, invert=invert)

# def visualize(image, heatmap, out_path=None, 
#               alpha=0.4, labels=None, width=20, aspect=0.2, 
#               hspace=0.2, cmap=shap.plots.colors.red_transparent_blue, invert=False):
#     if invert: image = -image
#     if len(heatmap.shape) == 3:
#         pixel_values = image.astype(np.float32)
#         pixel_values = (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min())

#         if not isinstance(heatmap, list):
#             heatmap = [heatmap]

#         if len(heatmap[0].shape) == 3:
#             heatmap = [v.reshape(1, *v.shape) for v in heatmap]
#             pixel_values = pixel_values.reshape(1, *pixel_values.shape)

#         if labels is not None:
#             if isinstance(labels, list):
#                 labels = np.array(labels).reshape(1, -1)

#         x = pixel_values
#         fig_size = np.array([3 * (len(heatmap) + 1), 1.5 * (x.shape[0] + 1)])
#         if fig_size[0] > width:
#             fig_size *= width / fig_size[0]
#         fig, axes = plt.subplots(nrows=x.shape[0], ncols=len(heatmap) + 1, figsize=fig_size)
#         if len(axes.shape) == 1:
#             axes = axes.reshape(1, axes.size)
#         for row in range(x.shape[0]):
#             x_curr = x[row].copy()

#             if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
#                 x_curr = x_curr.reshape(x_curr.shape[:2])

#             if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
#                 x_curr_gray = 0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2]
#                 x_curr_disp = x_curr
#             else:
#                 x_curr_gray = x_curr
#                 x_curr_disp = x_curr

#             axes[row, 0].imshow(x_curr_disp, cmap=plt.get_cmap('gray'))
#             axes[row, 0].axis('off')

#             if len(heatmap[0][row].shape) == 2:
#                 abs_vals = np.stack([np.abs(heatmap[i]) for i in range(len(heatmap))], 0).flatten()
#             else:
#                 abs_vals = np.stack([np.abs(heatmap[i].sum(-1)) for i in range(len(heatmap))], 0).flatten()
#             max_val = np.nanpercentile(abs_vals, 99.9)

#             for i in range(len(heatmap)):
#                 if labels is not None:
#                     axes[row, i + 1].set_title(labels[row, i])
#                 sv = heatmap[i][row] if len(heatmap[i][row].shape) == 2 else heatmap[i][row].sum(-1)
#                 axes[row, i + 1].imshow(x_curr_gray, cmap=plt.get_cmap('gray'), alpha=alpha, extent=(-1, sv.shape[1], sv.shape[0], -1))
#                 im = axes[row, i + 1].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
#                 axes[row, i + 1].axis('off')

#         if hspace == 'auto':
#             fig.tight_layout()
#         else:
#             fig.subplots_adjust(hspace=hspace)
#         cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0] / aspect)
#         cb.outline.set_visible(False)

#         if out_path is None:
#             plt.show()
#             # fig.canvas.draw()
#             # pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
#             # plt.close(fig)
#             # pil_image.show()
#         else:
#             fig.savefig(out_path)

#     else:
#         if image.shape[-1] == 1:
#             image = np.concatenate((image,) * 3, axis=-1)

#         heatmap = np.uint8(heatmap * 255)
#         jet = plt.get_cmap("jet")
#         jet_colors = jet(np.arange(256))[:, :3]
#         jet_heatmap = jet_colors[heatmap] * 255

#         si_img = jet_heatmap * alpha + (1 - alpha) * image
#         si_img = si_img.astype(np.uint8)
#         pil_img = Image.fromarray(si_img)
#         if out_path is None:
#             plt.imshow(si_img)
#             plt.axis('off')  # Hide the axis
#             plt.show()
#             # pil_img.show()
#         else:
#             pil_img.save(out_path)