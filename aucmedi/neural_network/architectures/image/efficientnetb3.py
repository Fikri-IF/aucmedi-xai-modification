#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
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
#                    Documentation                    #
#-----------------------------------------------------#
""" The classification variant of the EfficientNetB3 architecture.

| Architecture Variable    | Value                      |
| ------------------------ | -------------------------- |
| Key in architecture_dict | "2D.EfficientNetB3"        |
| Input_shape              | (300, 300)                 |
| Standardization          | "caffe"                    |

???+ abstract "Reference - Implementation"
    https://keras.io/api/applications/efficientnet/ <br>

???+ abstract "Reference - Publication"
    Mingxing Tan, Quoc V. Le. 28 May 2019.
    EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
    <br>
    https://arxiv.org/abs/1905.11946
"""
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.applications import EfficientNetB3
# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base

#-----------------------------------------------------#
#          Architecture class: EfficientNetB3         #
#-----------------------------------------------------#
class Architecture_EfficientNetB3(Architecture_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, classification_head, channels, input_shape=(300, 300),
                 pretrained_weights=False):
        self.classifier = classification_head
        self.input = input_shape + (channels,)
        self.pretrained_weights = pretrained_weights

    #---------------------------------------------#
    #                Create Model                 #
    #---------------------------------------------#
    def create_model(self, n_labels, fcl_dropout=True, activation_output="softmax",
                     pretrained_weights=False):
        # Get pretrained image weights from imagenet if desired
        if self.pretrained_weights : model_weights = "imagenet"
        else : model_weights = None

        # Obtain EfficientNet as base model
        base_model = EfficientNetB3(include_top=False, weights=model_weights,
                                    input_tensor=None, input_shape=self.input,
                                    pooling=None)
        top_model = base_model.output

        # Add classification head as top model
        
        
            
            
        
        

        
        model = Model(inputs=base_model.input, outputs=top_model)

        # Return created model
        return model
