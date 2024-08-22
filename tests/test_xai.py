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
#External libraries
import unittest
import tempfile
import os
import numpy as np
from PIL import Image
#Internal libraries
from aucmedi import *
from aucmedi.xai.decoder import *
from aucmedi.data_processing.io_loader import image_loader


#-----------------------------------------------------#
#              Unittest: Explainable AI               #
#-----------------------------------------------------#
class xaiTEST(unittest.TestCase):
    # Setup AUCMEDI pipeline
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data")
        # Create RGB data
        self.sampleList = []
        raw_background = []

        # Total Class
        self.total_class = 4

        for i in range(0, 10):
            img_rgb = np.random.rand(32, 32, 3) * 255
            raw_background.append(img_rgb.astype(np.uint8))
            img_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "image.sample_" + str(i) + ".RGB.png"
            path_sample = os.path.join(self.tmp_data.name, index)
            img_pillow.save(path_sample)
            self.sampleList.append(index)
        
        # Create classification labels
        self.labels_ohe = np.zeros((10, self.total_class), dtype=np.uint8)
        for i in range(0, 10):
            class_index = np.random.randint(0, self.total_class)
            self.labels_ohe[i][class_index] = 1
        
        # convert list to numpy
        self.background = np.array(raw_background)

        # Create class names
        self.class_names = ["class_0", "class_1", "class_2", "class_3"]

        # Define num eval
        self.num_eval = 10

        # Create Data Generator
        self.datagen = DataGenerator(self.sampleList,  self.tmp_data.name,
                                     labels=self.labels_ohe, resize=None,
                                     grayscale=False, batch_size=3)
        # Create Neural Network model
        self.model = NeuralNetwork(n_labels=self.total_class, channels=3, 
                                   input_shape=(32,32),architecture="2D.Vanilla", 
                                   batch_queue_size=1)
        # Compute predictions
        self.preds = self.model.predict(self.datagen)

        # Initialize testing image
        self.image = self.datagen[0][0][[0]]

        # Dummy XAI method instance for XAIDecoder testing
        self.xai_method = GradCAM(self.model.model)

    #-------------------------------------------------#
    #             XAI Functions: Decoder              #
    #-------------------------------------------------#
    def test_Decoder_Init(self):
        xai_decoder = XAIDecoder(self.datagen, self.model, self.xai_method, class_names=self.class_names, preds=self.preds)
        self.assertIsInstance(xai_decoder, XAIDecoder)
    
    def test_Decoder_Setter(self):
        xai_decoder = XAIDecoder(self.datagen, self.model, self.xai_method, class_names=self.class_names, preds=self.preds)
        new_method = Shap(self.model.model)
        xai_decoder.set_method(new_method)
        self.assertIsInstance(xai_decoder.get_method(),XAImethod_Base)
    
    def test_Decoder_argmax_output(self):
        xai_decoder = XAIDecoder(self.datagen, self.model, self.xai_method, class_names=self.class_names, preds=self.preds)
        xai_decoder.explain()

        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )
    
    def test_Decoder_allClasses_output(self):
        xai_decoder = XAIDecoder(self.datagen, self.model, self.xai_method, class_names=self.class_names)
        xai_decoder.explain()

        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )

    def test_Decoder_argmax_visualize(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_decoder = XAIDecoder(self.datagen, self.model, self.xai_method, class_names=self.class_names, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
            img = image_loader(sample=self.sampleList[i],
                               path_imagedir=self.tmp_data.name,
                               image_format=self.datagen.image_format)
            hm = image_loader(sample=file_xai,
                              path_imagedir=self.tmp_data.name,
                              image_format=self.datagen.image_format)
            self.assertTrue(np.array_equal(img.shape, hm.shape))
            self.assertFalse(np.array_equal(img, hm))
    
    def test_Decoder_allClasses_visualize(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_decoder = XAIDecoder(self.datagen, self.model, self.xai_method, class_names=self.class_names)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
                img = image_loader(sample=self.sampleList[i],
                                   path_imagedir=self.tmp_data.name,
                                   image_format=self.datagen.image_format)
                hm = image_loader(sample=file_xai,
                                  path_imagedir=self.tmp_data.name,
                                  image_format=self.datagen.image_format)
                self.assertTrue(np.array_equal(img.shape, hm.shape))
                self.assertFalse(np.array_equal(img, hm))

    #-------------------------------------------------#
    #            XAI Visualization: Heatmap           #
    #-------------------------------------------------#
    def test_Visualizer(self):
        image = self.image[0]
        image = np.expand_dims(image,axis=(0))
        heatmap = np.random.rand(32, 32)
        heatmap = np.expand_dims(heatmap, axis=(0,1))
        labels = np.array([self.class_names])
        path_xai = os.path.join(self.tmp_data.name)

        xai_decoder = XAIDecoder(self.datagen, self.model, self.xai_method, class_names=self.class_names)
        xai_decoder.results = heatmap
        xai_decoder.images = image
        xai_decoder.labels_result = labels

        xai_decoder.visualize_xai(out_path=path_xai, alpha=0.4)
        file_name = labels[0][np.argmax(self.preds[0])] + "_" + self.sampleList[0]
        self.assertTrue(os.path.exists(path_xai))
        img = image_loader(sample=self.sampleList[0],
                           path_imagedir=self.tmp_data.name,
                           image_format=self.datagen.image_format)
        hm = image_loader(sample=file_name,
                          path_imagedir=self.tmp_data.name,
                          image_format=None)
        self.assertTrue(np.array_equal(img.shape, hm.shape))
        self.assertFalse(np.array_equal(img, hm))

    #-------------------------------------------------#
    #              XAI Methods: Grad-Cam              #
    #-------------------------------------------------#
    def test_XAImethod_GradCam_init(self):
        GradCAM(self.model.model)

    def test_XAImethod_GradCam_heatmap(self):
        xai_method = GradCAM(self.model.model)
        for i in range(self.total_class):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (2,2)))

    def test_XAImethod_GradCam_decoder_Preds(self):
        xai_method = GradCAM(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()

        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )

    def test_XAImethod_GradCam_decoder_NoPreds(self):
        xai_method = GradCAM(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()

        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )

    def test_XAImethod_GradCam_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = GradCAM(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
            img = image_loader(sample=self.sampleList[i],
                               path_imagedir=self.tmp_data.name,
                               image_format=self.datagen.image_format)
            hm = image_loader(sample=file_xai,
                              path_imagedir=self.tmp_data.name,
                              image_format=self.datagen.image_format)
            self.assertTrue(np.array_equal(img.shape, hm.shape))
            self.assertFalse(np.array_equal(img, hm))
    
    def test_XAImethod_GradCam_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = GradCAM(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
                img = image_loader(sample=self.sampleList[i],
                                   path_imagedir=self.tmp_data.name,
                                   image_format=self.datagen.image_format)
                hm = image_loader(sample=file_xai,
                                  path_imagedir=self.tmp_data.name,
                                  image_format=self.datagen.image_format)
                self.assertTrue(np.array_equal(img.shape, hm.shape))
                self.assertFalse(np.array_equal(img, hm))
                        
    # #-------------------------------------------------#
    # #             XAI Methods: Grad-Cam++             #
    # #-------------------------------------------------#
    def test_XAImethod_GradCamPP_init(self):
        GradCAMpp(self.model.model)

    def test_XAImethod_GradCamPP_heatmap(self):
        xai_method = GradCAMpp(self.model.model)
        for i in range(self.total_class):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (2,2)))

    def test_XAImethod_GradCamPP_decoder_Preds(self):
        xai_method = GradCAMpp(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()

        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )

    def test_XAImethod_GradCamPP_decoder_NoPreds(self):
        xai_method = GradCAMpp(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()

        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )
    
    def test_XAImethod_GradCamPP_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = GradCAMpp(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
            img = image_loader(sample=self.sampleList[i],
                               path_imagedir=self.tmp_data.name,
                               image_format=self.datagen.image_format)
            hm = image_loader(sample=file_xai,
                              path_imagedir=self.tmp_data.name,
                              image_format=self.datagen.image_format)
            self.assertTrue(np.array_equal(img.shape, hm.shape))
            self.assertFalse(np.array_equal(img, hm))
    
    def test_XAImethod_GradCamPP_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = GradCAMpp(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
                img = image_loader(sample=self.sampleList[i],
                                   path_imagedir=self.tmp_data.name,
                                   image_format=self.datagen.image_format)
                hm = image_loader(sample=file_xai,
                                  path_imagedir=self.tmp_data.name,
                                  image_format=self.datagen.image_format)
                self.assertTrue(np.array_equal(img.shape, hm.shape))
                self.assertFalse(np.array_equal(img, hm))

    #-------------------------------------------------#
    #            XAI Methods: Saliency Maps           #
    #-------------------------------------------------#
    def test_XAImethod_SaliencyMap_init(self):
        SaliencyMap(self.model.model)

    def test_XAImethod_SaliencyMap_heatmap(self):
        xai_method = SaliencyMap(self.model.model)
        for i in range(self.total_class):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))

    def test_XAImethod_SaliencyMap_decoder_Preds(self):
        xai_method = SaliencyMap(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )

    def test_XAImethod_SaliencyMap_decoder_NoPreds(self):
        xai_method = SaliencyMap(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )
    
    def test_XAImethod_SaliencyMap_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = SaliencyMap(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
            img = image_loader(sample=self.sampleList[i],
                               path_imagedir=self.tmp_data.name,
                               image_format=self.datagen.image_format)
            hm = image_loader(sample=file_xai,
                              path_imagedir=self.tmp_data.name,
                              image_format=self.datagen.image_format)
            self.assertTrue(np.array_equal(img.shape, hm.shape))
            self.assertFalse(np.array_equal(img, hm))
    
    def test_XAImethod_SaliencyMap_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = SaliencyMap(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
                img = image_loader(sample=self.sampleList[i],
                                   path_imagedir=self.tmp_data.name,
                                   image_format=self.datagen.image_format)
                hm = image_loader(sample=file_xai,
                                  path_imagedir=self.tmp_data.name,
                                  image_format=self.datagen.image_format)
                self.assertTrue(np.array_equal(img.shape, hm.shape))
                self.assertFalse(np.array_equal(img, hm))


    #-------------------------------------------------#
    #       XAI Methods: Guided Backpropagation       #
    #-------------------------------------------------#
    def test_XAImethod_GuidedBackprop_init(self):
        GuidedBackpropagation(self.model.model)

    def test_XAImethod_GuidedBackprop_heatmap(self):
        xai_method = GuidedBackpropagation(self.model.model)
        for i in range(self.total_class):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))
        
    def test_XAImethod_GuidedBackprop_decoder_Preds(self):
        xai_method = GuidedBackpropagation(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )
    
    def test_XAImethod_GuidedBackprop_decoder_NoPreds(self):
        xai_method = GuidedBackpropagation(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )


    def test_XAImethod_GuidedBackprop_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = GuidedBackpropagation(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
            img = image_loader(sample=self.sampleList[i],
                               path_imagedir=self.tmp_data.name,
                               image_format=self.datagen.image_format)
            hm = image_loader(sample=file_xai,
                              path_imagedir=self.tmp_data.name,
                              image_format=self.datagen.image_format)
            self.assertTrue(np.array_equal(img.shape, hm.shape))
            self.assertFalse(np.array_equal(img, hm))
    
    def test_XAImethod_GuidedBackprop_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = GuidedBackpropagation(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
                img = image_loader(sample=self.sampleList[i],
                                   path_imagedir=self.tmp_data.name,
                                   image_format=self.datagen.image_format)
                hm = image_loader(sample=file_xai,
                                  path_imagedir=self.tmp_data.name,
                                  image_format=self.datagen.image_format)
                self.assertTrue(np.array_equal(img.shape, hm.shape))
                self.assertFalse(np.array_equal(img, hm))

    #-------------------------------------------------#
    #        XAI Methods: Integrated Gradients        #
    #-------------------------------------------------#
    def test_XAImethod_IntegratedGradients_init(self):
        IntegratedGradients(self.model.model, num_eval=self.num_eval)

    def test_XAImethod_IntegratedGradients_heatmap(self):
        xai_method = IntegratedGradients(self.model.model, num_eval=self.num_eval)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))

    def test_XAImethod_IntegratedGradients_decoder_Preds(self):
        xai_method = IntegratedGradients(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )
    
    def test_XAImethod_IntegratedGradients_decoder_NoPreds(self):
        xai_method = IntegratedGradients(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )
    
    def test_XAImethod_IntegratedGradients_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = IntegratedGradients(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
            img = image_loader(sample=self.sampleList[i],
                               path_imagedir=self.tmp_data.name,
                               image_format=self.datagen.image_format)
            hm = image_loader(sample=file_xai,
                              path_imagedir=self.tmp_data.name,
                              image_format=self.datagen.image_format)
            self.assertTrue(np.array_equal(img.shape, hm.shape))
            self.assertFalse(np.array_equal(img, hm))
    
    def test_XAImethod_IntegratedGradients_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = IntegratedGradients(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
                img = image_loader(sample=self.sampleList[i],
                                   path_imagedir=self.tmp_data.name,
                                   image_format=self.datagen.image_format)
                hm = image_loader(sample=file_xai,
                                  path_imagedir=self.tmp_data.name,
                                  image_format=self.datagen.image_format)
                self.assertTrue(np.array_equal(img.shape, hm.shape))
                self.assertFalse(np.array_equal(img, hm))

    #-------------------------------------------------#
    #           XAI Methods: Guided Grad-CAM          #
    #-------------------------------------------------#
    def test_XAImethod_GuidedGradCAM_init(self):
        GuidedGradCAM(self.model.model)

    def test_XAImethod_GuidedGradCAM_heatmap(self):
        xai_method = GuidedGradCAM(self.model.model)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))

    def test_XAImethod_GuidedGradCAM_decoder_Preds(self):
        xai_method = GuidedGradCAM(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )
    
    def test_XAImethod_GuidedGradCAM_decoder_NoPreds(self):
        xai_method = GuidedGradCAM(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )
    
    def test_XAImethod_GuidedGradCAM_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = GuidedGradCAM(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
            img = image_loader(sample=self.sampleList[i],
                               path_imagedir=self.tmp_data.name,
                               image_format=self.datagen.image_format)
            hm = image_loader(sample=file_xai,
                              path_imagedir=self.tmp_data.name,
                              image_format=self.datagen.image_format)
            self.assertTrue(np.array_equal(img.shape, hm.shape))
            self.assertFalse(np.array_equal(img, hm))
    
    def test_XAImethod_GuidedGradCAM_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = GuidedGradCAM(self.model.model)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
                img = image_loader(sample=self.sampleList[i],
                                   path_imagedir=self.tmp_data.name,
                                   image_format=self.datagen.image_format)
                hm = image_loader(sample=file_xai,
                                  path_imagedir=self.tmp_data.name,
                                  image_format=self.datagen.image_format)
                self.assertTrue(np.array_equal(img.shape, hm.shape))
                self.assertFalse(np.array_equal(img, hm))


    #-------------------------------------------------#
    #        XAI Methods: Occlusion Sensitivity       #
    #-------------------------------------------------#
    def test_XAImethod_OcclusionSensitivity_init(self):
        OcclusionSensitivity(self.model.model, num_eval=self.num_eval)

    def test_XAImethod_OcclusionSensitivity_heatmap(self):
        xai_method = OcclusionSensitivity(self.model.model, num_eval=self.num_eval)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))
    
    def test_XAImethod_OcclusionSensitivity_decoder_Preds(self):
        xai_method = OcclusionSensitivity(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )
    
    def test_XAImethod_OcclusionSensitivity_decoder_NoPreds(self):
        xai_method = OcclusionSensitivity(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )
    
    def test_XAImethod_OcclusionSensitivity_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = OcclusionSensitivity(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
            img = image_loader(sample=self.sampleList[i],
                               path_imagedir=self.tmp_data.name,
                               image_format=self.datagen.image_format)
            hm = image_loader(sample=file_xai,
                              path_imagedir=self.tmp_data.name,
                              image_format=self.datagen.image_format)
            self.assertTrue(np.array_equal(img.shape, hm.shape))
            self.assertFalse(np.array_equal(img, hm))
    
    def test_XAImethod_OcclusionSensitivity_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = OcclusionSensitivity(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
                img = image_loader(sample=self.sampleList[i],
                                   path_imagedir=self.tmp_data.name,
                                   image_format=self.datagen.image_format)
                hm = image_loader(sample=file_xai,
                                  path_imagedir=self.tmp_data.name,
                                  image_format=self.datagen.image_format)
                self.assertTrue(np.array_equal(img.shape, hm.shape))
                self.assertFalse(np.array_equal(img, hm))
                

    #-------------------------------------------------#
    #              XAI Methods: LIME Con              #
    #-------------------------------------------------#
    def test_XAImethod_LimeCon_init(self):
        LimeCon(self.model.model, num_eval=self.num_eval)

    def test_XAImethod_LimeCon_heatmap(self):
        xai_method = LimeCon(self.model.model, num_eval=self.num_eval)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))

    def test_XAImethod_LimeCon_decoder_Preds(self):
        xai_method = LimeCon(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )
    
    def test_XAImethod_LimeCon_decoder_NoPreds(self):
        xai_method = LimeCon(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )
    
    def test_XAImethod_LimeCon_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = LimeCon(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
            img = image_loader(sample=self.sampleList[i],
                               path_imagedir=self.tmp_data.name,
                               image_format=self.datagen.image_format)
            hm = image_loader(sample=file_xai,
                              path_imagedir=self.tmp_data.name,
                              image_format=self.datagen.image_format)
            self.assertTrue(np.array_equal(img.shape, hm.shape))
            self.assertFalse(np.array_equal(img, hm))
    
    def test_XAImethod_LimeCon_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = LimeCon(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
                img = image_loader(sample=self.sampleList[i],
                                   path_imagedir=self.tmp_data.name,
                                   image_format=self.datagen.image_format)
                hm = image_loader(sample=file_xai,
                                  path_imagedir=self.tmp_data.name,
                                  image_format=self.datagen.image_format)
                self.assertTrue(np.array_equal(img.shape, hm.shape))
                self.assertFalse(np.array_equal(img, hm))

    #-------------------------------------------------#
    #              XAI Methods: LIME Pro              #
    #-------------------------------------------------#
    def test_XAImethod_LimePro_init(self):
        LimePro(self.model.model, num_eval=self.num_eval)

    def test_XAImethod_LimePro_heatmap(self):
        xai_method = LimePro(self.model.model, num_eval=self.num_eval)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))

    def test_XAImethod_LimePro_decoder_Preds(self):
        xai_method = LimePro(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )
    
    def test_XAImethod_LimePro_decoder_NoPreds(self):
        xai_method = LimePro(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )
    
    def test_XAImethod_LimePro_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = LimePro(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
            img = image_loader(sample=self.sampleList[i],
                               path_imagedir=self.tmp_data.name,
                               image_format=self.datagen.image_format)
            hm = image_loader(sample=file_xai,
                              path_imagedir=self.tmp_data.name,
                              image_format=self.datagen.image_format)
            self.assertTrue(np.array_equal(img.shape, hm.shape))
            self.assertFalse(np.array_equal(img, hm))
    
    def test_XAImethod_LimePro_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = LimePro(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
                img = image_loader(sample=self.sampleList[i],
                                   path_imagedir=self.tmp_data.name,
                                   image_format=self.datagen.image_format)
                hm = image_loader(sample=file_xai,
                                  path_imagedir=self.tmp_data.name,
                                  image_format=self.datagen.image_format)
                self.assertTrue(np.array_equal(img.shape, hm.shape))
                self.assertFalse(np.array_equal(img, hm))

    #-------------------------------------------------#
    #              XAI Methods: SHAP                  #
    #-------------------------------------------------#
    def test_XAImethod_Shap_init(self):
        Shap(self.model.model, num_eval=self.num_eval)
    
    def test_XAImethod_Shap_heatmap(self):
        xai_method = Shap(self.model.model, num_eval=self.num_eval)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32,3)))
    
    def test_XAImethod_Shap_decoder_Preds(self):
        xai_method = Shap(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32,3)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )
    
    def test_XAImethod_Shap_decoder_NoPreds(self):
        xai_method = Shap(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32, 3)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )
    
    def test_XAImethod_Shap_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = Shap(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
    
    def test_XAImethod_Shap_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = Shap(self.model.model, num_eval=self.num_eval)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
    
    #-------------------------------------------------#
    #              XAI Methods: DeepSHAP              #
    #-------------------------------------------------#
    def test_XAImethod_DeepShap_init(self):
        DeepShap(self.model.model, background=self.background)

    def test_XAImethod_DeepShap_heatmap(self):
        xai_method = DeepShap(self.model.model, background=self.background)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32,3)))

    def test_XAImethod_DeepShap_decoder_Preds(self):
        xai_method = DeepShap(self.model.model, background=self.background)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32,3)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )
            
    def test_XAImethod_DeepShap_decoder_NoPreds(self):
        xai_method = DeepShap(self.model.model, background=self.background)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32, 3)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )
    
    def test_XAImethod_DeepShap_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = DeepShap(self.model.model, background=self.background)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
    
    def test_XAImethod_DeepShap_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = DeepShap(self.model.model, background=self.background)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
    
    #-------------------------------------------------#
    #           XAI Methods: GradientSHAP             #
    #-------------------------------------------------#
    def test_XAImethod_GradientShap_init(self):
        GradientShap(self.model.model, background=self.background)

    def test_XAImethod_GradientShap_heatmap(self):
        xai_method = GradientShap(self.model.model, background=self.background)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32,3)))
    
    def test_XAImethod_GradientShap_decoder_Preds(self):
        xai_method = GradientShap(self.model.model, background=self.background)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32,3)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )
    
    def test_XAImethod_GradientShap_decoder_NoPreds(self):
        xai_method = GradientShap(self.model.model, background=self.background)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32, 3)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )

    def test_XAImethod_GradientShap_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = GradientShap(self.model.model, background=self.background)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
        
    def test_XAImethod_GradientShap_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = GradientShap(self.model.model, background=self.background)
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))

    #-------------------------------------------------#
    #              XAI Methods: RISE                  #
    #-------------------------------------------------#
    def test_XAImethod_Rise_init(self):
        Rise(self.model.model, total_mask=1000, mask_size=8, mode='blur')
    
    def test_XAImethod_Rise_heatmap(self):
        xai_method = Rise(self.model.model, total_mask=1000, mask_size=8, mode='blur')
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))
    
    def test_XAImethod_Rise_decoder_Preds(self):
        xai_method = Rise(self.model.model, total_mask=1000, mask_size=8, mode='blur')
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, 1, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, 1))
        )
    
    def test_XAImethod_Rise_decoder_NoPreds(self):
        xai_method = Rise(self.model.model, total_mask=1000, mask_size=8, mode='blur')
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        self.assertTrue(
            np.array_equal(xai_decoder.results.shape, (10, self.total_class, 32, 32)) and
            np.array_equal(xai_decoder.images.shape, (10, 32, 32,3)) and
            np.array_equal(xai_decoder.labels_result.shape,(10, self.total_class))
        )
    
    def test_XAImethod_Rise_visualize_Preds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = Rise(self.model.model, total_mask=1000, mask_size=8, mode='blur')
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method, preds=self.preds)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            file_xai = self.class_names[np.argmax(self.preds[i])] + "_" + self.sampleList[i]
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                         file_xai)
            self.assertTrue(os.path.exists(path_xai_file))
            img = image_loader(sample=self.sampleList[i],
                               path_imagedir=self.tmp_data.name,
                               image_format=self.datagen.image_format)
            hm = image_loader(sample=file_xai,
                              path_imagedir=self.tmp_data.name,
                              image_format=self.datagen.image_format)
            self.assertTrue(np.array_equal(img.shape, hm.shape))
            self.assertFalse(np.array_equal(img, hm))
    
    def test_XAImethod_Rise_visualize_NoPreds(self):
        path_xai = os.path.join(self.tmp_data.name)

        xai_method = Rise(self.model.model, total_mask=1000, mask_size=8, mode='blur')
        xai_decoder = XAIDecoder(self.datagen, self.model, xai_method)
        xai_decoder.explain()
        xai_decoder.visualize_xai(out_path=path_xai)

        for i in range(0, len(self.sampleList)):
            for j in range(0, self.total_class):
                file_xai = self.class_names[j] + "_" + self.sampleList[i]
                path_xai_file = os.path.join(os.path.join(self.tmp_data.name),
                                             file_xai)
                self.assertTrue(os.path.exists(path_xai_file))
                img = image_loader(sample=self.sampleList[i],
                                   path_imagedir=self.tmp_data.name,
                                   image_format=self.datagen.image_format)
                hm = image_loader(sample=file_xai,
                                  path_imagedir=self.tmp_data.name,
                                  image_format=self.datagen.image_format)
                self.assertTrue(np.array_equal(img.shape, hm.shape))
                self.assertFalse(np.array_equal(img, hm))

if __name__ == '__main__':
#     runner = HTMLTestRunner(
#         report_filepath="xai_test.html",
#         title="XAI functionalities unit test",
#         description="This demonstrates the report output by HTMLTestRunner.",
#         open_in_browser=True
#     )
    unittest.main()
    # xaiTEST.setUpClass()
    # xaiTEST.test_XAImethod_Rise_init(xaiTEST)
    # xaiTEST.test_XAImethod_Rise_heatmap(xaiTEST)