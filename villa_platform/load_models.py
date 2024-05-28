from transformers import (
    AutoConfig,
    PretrainedConfig
)
from llava_llama import LlavaLlamaModel
import torch
from constants import *
class LoadVilaImage:
    def __int__(self,model_path='',model_name=''):
        self.model_path = model_path
        self.model_name = model_name
    def load_pretrained_model(
            self,
            model_path,
            device="cuda",
            **kwargs,
    ):
        #By default it will load Llavallama
        config = AutoConfig.from_pretrained(model_path)
        print(f"INFO: LOADED CONFIG,This is model config",config)
        config.resume_path = model_path
        self.prepare_config_for_eval(config, kwargs)
        model = LlavaLlamaModel(
            config=config,
            low_cpu_mem_usage=True,
            **kwargs
        )

        tokenizer = model.tokenizer
        model.eval()
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        vision_tower.to(device=device, dtype=torch.float16)
        mm_projector = model.get_mm_projector()
        mm_projector.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor
        if hasattr(model.llm.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        return tokenizer, model, image_processor, context_len
    def prepare_config_for_eval(self,config: PretrainedConfig, kwargs: dict):
        try:
            # compatible with deprecated config convention
            if getattr(config, "vision_tower_cfg", None) is None:
                config.vision_tower_cfg = config.mm_vision_tower
        except AttributeError:
            raise ValueError(f"Invalid configuration! Cannot find vision_tower in config:\n{config}")

        config.model_dtype = kwargs.pop("torch_dtype").__str__()
        # siglip does not support device_map = "auto"
        vision_tower_name = self.parse_model_name_or_path(config, "vision_tower")
        if "siglip" in vision_tower_name.lower():
            kwargs["device_map"] = "cuda"

    def parse_model_name_or_path(self,config: PretrainedConfig, model_name="llm", suffix="_cfg"):
        target_model = f"{model_name}{suffix}"
        target_cfg = getattr(config, target_model, None)

        if isinstance(target_cfg, str):
            return target_cfg
        elif isinstance(target_cfg, dict):
            return target_cfg["architectures"][0]
        else:
            raise ValueError(f"Invalid {target_model} configuration!")