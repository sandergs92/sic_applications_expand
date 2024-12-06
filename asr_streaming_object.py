import copy
import time
import pyaudio as pa
import numpy as np
import torch

from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

class ASRStreaming:
    def __init__(self, model_name="stt_en_fastconformer_hybrid_large_streaming_multi", lookahead_size=80, decoder_type="rnnt"):
        self.SAMPLE_RATE = 16000
        self.ENCODER_STEP_LENGTH = 80
        self.model_name = model_name
        self.lookahead_size = lookahead_size
        self.decoder_type = decoder_type

        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        self._validate_and_configure_model()

        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = self.asr_model.encoder.get_initial_cache_state(batch_size=1)
        self.previous_hypotheses = None
        self.pred_out_stream = None
        self.step_num = 0
        self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
        self.num_channels = self.asr_model.cfg.preprocessor.features
        self.cache_pre_encode = torch.zeros((1, self.num_channels, self.pre_encode_cache_size), device=self.asr_model.device)

        self.preprocessor = self._init_preprocessor()

    def _validate_and_configure_model(self):
        if self.model_name == "stt_en_fastconformer_hybrid_large_streaming_multi":
            if self.lookahead_size not in [0, 80, 480, 1040]:
                raise ValueError(
                    f"Specified lookahead_size {self.lookahead_size} is not valid. Allowed values: 0, 80, 480, 1040."
                )
            left_context_size = self.asr_model.encoder.att_context_size[0]
            self.asr_model.encoder.set_default_att_context_size([left_context_size, int(self.lookahead_size / self.ENCODER_STEP_LENGTH)])

        self.asr_model.change_decoding_strategy(decoder_type=self.decoder_type)
        decoding_cfg = self.asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = False
            if hasattr(self.asr_model, 'joint'):
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
            self.asr_model.change_decoding_strategy(decoding_cfg)

        self.asr_model.eval()

    def _init_preprocessor(self):
        cfg = copy.deepcopy(self.asr_model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        preprocessor.to(self.asr_model.device)
        return preprocessor

    def _preprocess_audio(self, audio):
        device = self.asr_model.device
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        return processed_signal, processed_signal_length

    def _extract_transcriptions(self, hyps):
        if isinstance(hyps[0], Hypothesis):
            return [hyp.text for hyp in hyps]
        return hyps

    def transcribe_chunk(self, new_chunk):
        audio_data = new_chunk.astype(np.float32) / 32768.0
        processed_signal, processed_signal_length = self._preprocess_audio(audio_data)
        processed_signal = torch.cat([self.cache_pre_encode, processed_signal], dim=-1)
        processed_signal_length += self.cache_pre_encode.shape[1]
        self.cache_pre_encode = processed_signal[:, :, -self.pre_encode_cache_size:]

        with torch.no_grad():
            (
                self.pred_out_stream,
                transcribed_texts,
                self.cache_last_channel,
                self.cache_last_time,
                self.cache_last_channel_len,
                self.previous_hypotheses,
            ) = self.asr_model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=self.previous_hypotheses,
                previous_pred_out=self.pred_out_stream,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )

        final_streaming_tran = self._extract_transcriptions(transcribed_texts)
        self.step_num += 1
        return final_streaming_tran[0]
    
    def reset_hypothesis(self):
        # init params we will use for streaming
        self.previous_hypotheses = None
        self.pred_out_stream = None
        self.step_num = 0
        self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
        # cache-aware models require some small section of the previous processed_signal to
        # be fed in at each timestep - we initialize this to a tensor filled with zeros
        # so that we will do zero-padding for the very first chunk(s)
        self.num_channels = self.asr_model.cfg.preprocessor.features
        self.cache_pre_encode = torch.zeros((1, self.num_channels, self.pre_encode_cache_size), device=self.asr_model.device)