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
import os
# AUCMEDI Libraries
from aucmedi.data_processing.io_loader import image_loader
from aucmedi.data_processing.subfunctions import Resize
from aucmedi.data_processing.data_generator import DataGenerator
from aucmedi.neural_network.model import NeuralNetwork
from aucmedi.xai.methods import *

class XAIDecoder():
    def __init__(self, data_gen : DataGenerator,
                 model : NeuralNetwork, 
                 method : XAImethod_Base,
                 class_names=None,
                 preds=None):
        
        self.__method = method
        self.data_gen = data_gen
        self.model = model
        self.preds = preds
        self.n_classes = model.n_labels
        self.sample_list = data_gen.samples
        self.labels = class_names if class_names is not None else ["Class_" + str(i) for i in range(0, self.n_classes)]

        self.results = []
        self.images = []
        self.labels_result = []
        self.samples = np.array(self.sample_list)
    
    def explain(self):
        for i, sample in enumerate(self.sample_list):
            img_batch, shape_org, img_org = self.__load_and_preprocess_image(sample, i)
            
            if self.preds is not None:
                class_index = np.argmax(self.preds[i])
                result = self.__process_preds(img_batch, shape_org, class_index)
                self.labels_result.append([self.labels[class_index]])
            else:
                result = self.__process_all_classes(img_batch, shape_org)
                self.labels_result.append(self.labels)
            self.results.append(result)
            self.images.append(img_org)

        self.results = np.array(self.results)
        self.images = np.array(self.images)
        self.labels_result = np.array(self.labels_result)
    
    def get_method(self) -> XAImethod_Base :
        return self.__method
    
    def set_method(self, method : XAImethod_Base):
        self.__method = method

    def visualize_xai(self, alpha=0.4, invert=False, out_path=None):
        """
        Visualizes the XAI results.

        Parameters:
        - xai_result: XAIResult object containing images, xai_maps, labels, and samples.
        - alpha: float, opacity level for the XAI map overlay.
        - invert: bool, whether to invert the colors.
        - out_path: str, directory to save the visualized images.
        """

        num_classes = self.results.shape[1]
        if invert: self.images = -self.images

        for i in range(len(self.images)):
            for j in range(num_classes):
                file_name = None
                if out_path is not None:

                    os.makedirs(out_path, exist_ok=True)

                    file_name = f"{self.labels_result[i][j]}_{self.samples[i]}"
                    if os.sep in file_name : file_name = file_name.replace(os.sep, ".")
                    file_name = os.path.join(out_path, file_name)
                    
                self.__method.visualize_heatmap(self.images[i], self.results[i][j], out_path=file_name, alpha=alpha, labels=self.labels_result)
                
    def __load_and_preprocess_image(self, sample, index):
        img_org = image_loader(sample, self.data_gen.path_imagedir,
                               image_format=self.data_gen.image_format,
                               grayscale=self.data_gen.grayscale)
        shape_org = img_org.shape[0:2]
        img_prc = self.data_gen.preprocess_image(index)
        img_batch = np.expand_dims(img_prc, axis=0)
        return img_batch, shape_org, img_org

    def __process_preds(self, img_batch, shape_org, class_index):
        xai_map = self.__method.compute_heatmap(img_batch, class_index=class_index, all_class=False)
        xai_map = Resize(shape=shape_org).transform(xai_map)
        xai_maps = np.array([xai_map])
        return xai_maps

    def __process_all_classes(self, img_batch, shape_org):
        xai_maps = []
        for ci in range(0, self.n_classes):
            xai_map = self.__method.compute_heatmap(img_batch, class_index=ci, all_class=True)
            xai_map = Resize(shape=shape_org).transform(xai_map)
            xai_maps.append(xai_map)
        xai_maps = np.array(xai_maps)
        return xai_maps

#-----------------------------------------------------#
#                    XAI - Decoder                    #
#-----------------------------------------------------#
# def xai_decoder(data_gen, model, preds=None, method="gradcam", 
#                 layer_name=None, class_names=None, num_eval=None,
#                 background=None) -> XAIDecoder:

#     shap_based = False
#     if "shap" in str(xai_method).lower() :shap_based = True
#     if isinstance(method, str) and method in xai_dict:
#         xai_method = xai_dict[method](model.model, layerName=layer_name, num_eval=num_eval, background=background)
#     else: xai_method = method

#     return XAIDecoder(data_gen, model, xai_method, shap_based, preds)

#-----------------------------------------------------#
#                    XAI - Decoder                    #
#-----------------------------------------------------#
# def xai_decoder(data_gen, model, preds=None, method="gradcam", 
#                 layerName=None, classNames=None, num_eval=None,
#                 background=None,alpha=0.4, out_path=None):
#     """ XAI Decoder function for automatic computation of Explainable AI heatmaps.

#     This module allows to visualize which regions were crucial for the neural network model
#     to compute a classification on the provided unknown images.

#     - If `out_path` parameter is None, heatmaps are returned as NumPy array.
#     - If a path is provided as `out_path`, then heatmaps are stored to disk as PNG files.

#     ???+ info "XAI Methods"
#         The XAI Decoder can be run with different XAI methods as backbone.

#         A list of all implemented methods and their keys can be found here: <br>
#         [aucmedi.xai.methods][]

#     ???+ example "Example"
#         ```python
#         # Create a DataGenerator for data I/O
#         datagen = DataGenerator(samples[:3], "images_xray/", labels=None, resize=(299, 299))

#         # Get a model
#         model = NeuralNetwork(n_labels=3, channels=3, architecture="Xception",
#                                input_shape=(299,299))
#         model.load("model.xray.hdf5")

#         # Make some predictions
#         preds = model.predict(datagen)

#         # Compute XAI heatmaps via Grad-CAM (resulting heatmaps are stored in out_path)
#         xai_decoder(datagen, model, preds, method="gradcam", out_path="xai.xray_gradcam")
#         ```

#     Args:
#         data_gen (DataGenerator):           A data generator which will be used for inference.
#         model (NeuralNetwork):             Instance of a AUCMEDI neural network class.
#         preds (numpy.ndarray):              NumPy Array of classification prediction encoded as OHE (output of a AUCMEDI prediction).
#         method (str):                       XAI method class instance or index. By default, GradCAM is used as XAI method.
#         layerName (str):                    Layer name of the convolutional layer for heatmap computation. If `None`, the last conv layer is used.
#         alpha (float):                      Transparency value for heatmap overlap plotting on input image (range: [0-1]).
#         out_path (str):                     Output path in which heatmaps are saved to disk as PNG files.
#         iterations (int):                   Number of iterations for the XAI instance explanations.
#         invert (bool):                      Invert the input image for XAI heatmap visualization.

#     Returns:
#         images (numpy.ndarray):             Combined array of images. Will be only returned if `out_path` parameter is `None`.
#         heatmaps (numpy.ndarray):           Combined array of XAI heatmaps. Will be only returned if `out_path` parameter is `None`.
#     """
#     # Initialize & access some variables
#     batch_size = data_gen.batch_size
#     n_classes = model.n_labels
#     sample_list = data_gen.samples

#     # Prepare XAI output methods
#     res_img = []
#     res_xai = []
#     if out_path is not None and not os.path.exists(out_path) : os.mkdir(out_path)
#     # Initialize xai method
#     if isinstance(method, str) and method in xai_dict:
#         xai_method = xai_dict[method](model.model, layerName=layerName, num_eval=num_eval, background=background)
#     else : xai_method = method

#     # Check if SHAP method is used
#     shap_method = False
#     if "shap" in str(xai_method).lower() :shap_method = True

#     if classNames is None:
#         classNames = ["Class_" + str(i) for i in range(0, n_classes)]
    
#     # Iterate over all samples
#     for i in range(0, len(sample_list)):
#         # Load original image
#         img_org = image_loader(sample_list[i], data_gen.path_imagedir,
#                                image_format=data_gen.image_format,
#                                grayscale=data_gen.grayscale)
#         shape_org = img_org.shape[0:2]
#         # Load processed image
#         img_prc = data_gen.preprocess_image(i)
#         img_batch = np.expand_dims(img_prc, axis=0)

#         # Compute SHAP heatmap
#         if shap_method:
#             xai_map = xai_method.compute_heatmap(img_batch, class_index=None)
#             xai_map = Resize(shape=shape_org).transform(xai_map)
#             label = None
#             if preds is not None: label = get_prediction_label(preds[i], classNames=classNames)
#             visualize(image=img_org,shap_values=xai_map,labels=label,alpha=alpha)
#             continue
#         # If preds given, compute heatmap only for argmax class
#         if preds is not None:
#             ci = np.argmax(preds[i])
#             xai_map = xai_method.compute_heatmap(img_batch, class_index=ci)
#             xai_map = Resize(shape=shape_org).transform(xai_map)
#             postprocess_output(sample_list[i], img_org, xai_map, n_classes,
#                                data_gen, res_img, res_xai, out_path, alpha)
#         # If no preds given, compute heatmap for all classes
#         else:
#             sample_maps = []
#             for ci in range(0, n_classes):
#                 xai_map = xai_method.compute_heatmap(img_batch, class_index=ci)
#                 xai_map = Resize(shape=shape_org).transform(xai_map)
#                 sample_maps.append(xai_map)
#             sample_maps = np.array(sample_maps)
#             postprocess_output(sample_list[i], img_org, sample_maps, n_classes,
#                                data_gen, res_img, res_xai, out_path, alpha)
#     # Return output directly if no output path is defined (and convert to NumPy)
#     if out_path is None : return np.array(res_img), np.array(res_xai)


# """Helper. Get the class name for a specific class index.

# Args:
#     class_index (int): Classification index for which the class name is requested.
# """
# def get_prediction_label(class_index, classNames):

#     if len(class_index) > 1:
#         # For softmax output, return the class with the highest probability
#         final_index = np.argmax(class_index)
#     else:
#         # For sigmoid output
#         if class_index[0] >= 0.5:
#             final_index = 1
#         else:
#             final_index = 0
#     class_name = [[classNames[final_index]]]
#     return class_name
# #-----------------------------------------------------#
# #          Subroutine: Output Postprocessing          #
# #-----------------------------------------------------#
# """ Helper/Subroutine function for XAI Decoder.

# Caches heatmap for direct output or generates a visualization as PNG.
# """
# def postprocess_output(sample, image, xai_map, n_classes, data_gen,
#                        res_img, res_xai, out_path, alpha):
#     # Update result lists for direct output
#     if out_path is None:
#         res_img.append(image)
#         res_xai.append(xai_map)
#     # Generate XAI heatmap visualization
#     else:
#         # Create XAI path
#         if data_gen.image_format:
#             xai_file = sample + "." + data_gen.image_format
#         else : xai_file = sample
#         if os.sep in xai_file : xai_file = xai_file.replace(os.sep, ".")
#         path_xai = os.path.join(out_path, xai_file)
#         # If preds given, output only argmax class heatmap
#         if len(xai_map.shape) == 2:
#             visualize(image=image, heatmap=xai_map, out_path=path_xai, alpha=alpha)
#         # If no preds given, output heatmaps for all classes
#         else:
#             for c in range(0, n_classes):
#                 path_xai_c = path_xai[:-4] + ".class_" + str(c) + \
#                              path_xai[-4:]
#                 visualize(image=image, heatmap=xai_map[c], out_path=path_xai_c,
#                                   alpha=alpha)