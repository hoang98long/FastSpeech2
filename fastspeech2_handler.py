import logging
from utils.tools import to_device
from matplotlib.pyplot import text
import numpy as np
import os
import torch
import uuid
import zipfile
# from waveglow_model import WaveGlow
from scipy.io.wavfile import write, read
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

from synthesize import preprocess_english, preprocess_mandarin, synthesize_wav
from .model.fastspeech2 import FastSpeech2
from .utils.model import get_model, get_vocoder
from .utils.tools import synth_wav
import yaml


preprocess_config = yaml.load(
    open( './config/Viet_tts/preprocess.yaml', "r"), Loader=yaml.FullLoader
)
model_config = yaml.load(open('./config/Viet_tts/model.yaml', "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open('./config/Viet_tts/train.yaml', "r"), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)

class Args:
	restore_step = 30000
	pitch_control = 1.
	energy_control = 1.
	duration_control = 1.

class FastSpeech2Synthesizer(BaseHandler):

	def __init__(self):
		super().__init__()
		self.model = None
		self.vocoder = None
		self.initialized = False
	
	def initialize(self, context):
		# return super().initialize(context)
		properties = context.system_properties
		model_dir = property.get("model_dir")
		# if not torch.cuda.is_available() or properties.get("gpu_id") is None:
		# 	raise RuntimeError("GPU required!")
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		args = Args()
		self.model = get_model(args, configs, self.device, train=False)

		self.vocoder = get_vocoder(model_config, self.device)
		self.model.eval()
		logger.debug("FastSpeech2 model file loaded successfully!")
		self.initialized = True
	
	def preprocess(self, data):
		# return super().preprocess(data)
		ids = raw_texts = data
		speakers = np.array([0])
		if preprocess_config["preprocessing"]["text"]["language"] == "en":
			texts = np.array([preprocess_english(text, preprocess_config)])
		elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
			texts = np.array([preprocess_mandarin(text, preprocess_config)])
		text_lens = np.array([len(texts[0])])
		batch = (ids, raw_texts, speakers, texts, text_lens, max(text_lens))
		return batch
	
	def inference(self, data, *args, **kwargs):
		# return super().inference(data, *args, **kwargs)
		control_values = 1., 1., 1.
		pitch_control, energy_control, duration_control = control_values
		batch = to_device(data, self.device)
		with torch.no_grad():
			output = self.model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
			)
			wav_file = synth_wav(
                batch,
                output,
                self.vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                )
		return wav_file
	
	# TODO: save inference data
	def postprocess(self, wav_file):
		# return super().postprocess(data)
		with open(wav_file, 'rb') as output:
			data = output.read()
		os.remove(wav_file)
		return [data]