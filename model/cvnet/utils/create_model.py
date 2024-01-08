# Copyright 2022 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn


def create_model(
        module: nn.Module,
        configs,
        pretrained: bool = False,
        checkpoint_path: str = None,
        device: str = None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = module(**configs).to(device)
    if pretrained:
        if checkpoint_path:
            local_path = checkpoint_path
        else:
            raise AttributeError('No url or checkpoint_path is provided for pretrained model.')
        state_dict = torch.load(local_path, map_location=device)
        try:
            state_dict = state_dict["model_state"]
        except KeyError:
            state_dict = state_dict

        model_dict = model.state_dict()
        state_dict = {k : v for k, v in state_dict.items()}
        weight_dict = {k : v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

        if len(weight_dict) == len(state_dict):
            print('All parameters are loaded')
        else:
            raise AssertionError("The model is not fully loaded.")

        model_dict.update(weight_dict)
        model.load_state_dict(model_dict)

    model.eval()
    return model
