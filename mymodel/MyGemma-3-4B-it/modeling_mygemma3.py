import json
import numpy as np
import torch
from transformers import Gemma3ForConditionalGeneration


class MyGemma3ForConditionalGeneration(Gemma3ForConditionalGeneration):
    def __init__(self, config, steering_vec_file=None, control_type="none",
                 control_num=0, control_scale=1):
        super().__init__(config)
        self.steering_vec_layer = 0
        self.steering_vec_type = "none"
        self.steering_vec_dict = None
        self.steering_vec_scale = 1

        if steering_vec_file is not None and control_type != "none":
            hidden_size = config.text_config.hidden_size
            self.steering_vec_layer = int(control_num)
            self.steering_vec_type = control_type
            self.steering_vec_scale = float(control_scale)
            self.steering_vec_dict = {
                int(k): np.array(v).reshape(-1, hidden_size)
                for k, v in json.load(open(steering_vec_file)).items()
            }
            self._register_steering_hooks()

    def _make_hook(self, layer_id):
        def hook(module, input, output):
            if self.steering_vec_dict is None:
                return output
            hidden_states = output[0]
            if hidden_states.shape[1] > 1 and layer_id == self.steering_vec_layer:
                vec = torch.tensor(
                    self.steering_vec_dict[layer_id]
                ).to(hidden_states.device).to(torch.float)
                hidden_states = hidden_states + vec * self.steering_vec_scale
                return (hidden_states,) + output[1:]
            return output
        return hook

    def _register_steering_hooks(self):
        for layer_id, layer in enumerate(self.language_model.layers):
            layer.register_forward_hook(self._make_hook(layer_id))
