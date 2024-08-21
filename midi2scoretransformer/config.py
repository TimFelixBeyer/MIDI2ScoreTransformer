from transformers import RoFormerConfig


# TODO: Make this single source of truth
FEATURES = {
    "onset": {'vocab_size': 200, 'loss_weight': 1.0, 'ignore_index': -100, 'min': 0, 'max': 8, 'step_size': 1/24},
    "offset": {'vocab_size': 145, 'loss_weight': 0.25, 'ignore_index': -100, 'min': 0, 'max': 6, 'step_size': 1/24},
    "downbeat": {'vocab_size': 146, 'loss_weight': 0.4, 'ignore_index': -100, 'min': -1/24, 'max': 6, 'step_size': 1/24},
    "duration": {'vocab_size': 97, 'loss_weight': 0.3, 'ignore_index': -100, 'min': 0, 'max': 96},
    "pitch": {'vocab_size': 128, 'loss_weight': 1.0, 'ignore_index': -100, 'min': 0, 'max': 127},
    "accidental": {'vocab_size': 7, 'loss_weight': 0.5, 'ignore_index': 6, 'min': 0, 'max': 6},
    "keysignature": {'vocab_size': 16, 'loss_weight': 0.5, 'ignore_index': 15, 'min': 0, 'max': 15},
    "velocity": {'vocab_size': 8, 'loss_weight': 0.0, 'ignore_index': -100, 'min': 0, 'max': 127},
    "grace": {'vocab_size': 2, 'loss_weight': 1.0, 'ignore_index': -100, 'min': 0, 'max': 1},
    "trill": {'vocab_size': 2, 'loss_weight': 1.0, 'ignore_index': -100, 'min': 0, 'max': 1},
    "staccato": {'vocab_size': 2, 'loss_weight': 0.15, 'ignore_index': -100, 'min': 0, 'max': 1},
    "voice": {'vocab_size': 9, 'loss_weight': 0.3, 'ignore_index': 0, 'min': 0, 'max': 8},
    "stem": {'vocab_size': 4, 'loss_weight': 0.2, 'ignore_index': 3, 'min': 0, 'max': 3},
    "hand": {'vocab_size': 3, 'loss_weight': 0.25, 'ignore_index': 2, 'min': 0, 'max': 2},
}

class MyModelConfig(RoFormerConfig):
    def __init__(
        self,
        input_streams=4,
        in_onset_vocab_size=200,
        in_duration_vocab_size=200,
        in_pitch_vocab_size=128,
        in_velocity_vocab_size=8,
        out_offset_vocab_size=FEATURES['offset']['vocab_size'],
        out_downbeat_vocab_size=FEATURES['downbeat']['vocab_size'],
        out_duration_vocab_size=FEATURES['duration']['vocab_size'],
        out_pitch_vocab_size=FEATURES['pitch']['vocab_size'],
        out_accidental_vocab_size=FEATURES['accidental']['vocab_size'],  # need one class as ignore class for untagged inputs
        out_keysignature_vocab_size=FEATURES['keysignature']['vocab_size'],  # need one class as ignore class for untagged inputs
        out_velocity_vocab_size=FEATURES['velocity']['vocab_size'],
        out_grace_vocab_size=FEATURES['grace']['vocab_size'],
        out_trill_vocab_size=FEATURES['trill']['vocab_size'],
        out_staccato_vocab_size=FEATURES['staccato']['vocab_size'],
        out_voice_vocab_size=FEATURES['voice']['vocab_size'],  # need one class as ignore class for untagged inputs
        out_stem_vocab_size=FEATURES['stem']['vocab_size'],  # need one class as ignore class for untagged inputs
        out_hand_vocab_size=FEATURES['hand']['vocab_size'],  # need one class as ignore class for untagged inputs
        is_autoregressive=False,
        positional_encoding="RoPE",
        conditional_sampling=False,
        bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_streams = input_streams
        self.in_onset_vocab_size = in_onset_vocab_size
        self.in_duration_vocab_size = in_duration_vocab_size
        self.in_pitch_vocab_size = in_pitch_vocab_size
        self.in_velocity_vocab_size = in_velocity_vocab_size
        self.out_offset_vocab_size = out_offset_vocab_size
        self.out_downbeat_vocab_size = out_downbeat_vocab_size
        self.out_duration_vocab_size = out_duration_vocab_size
        self.out_pitch_vocab_size = out_pitch_vocab_size
        self.out_accidental_vocab_size = out_accidental_vocab_size
        self.out_keysignature_vocab_size = out_keysignature_vocab_size
        self.out_velocity_vocab_size = out_velocity_vocab_size
        self.out_grace_vocab_size = out_grace_vocab_size
        self.out_trill_vocab_size = out_trill_vocab_size
        self.out_staccato_vocab_size = out_staccato_vocab_size
        self.out_voice_vocab_size = out_voice_vocab_size
        self.out_stem_vocab_size = out_stem_vocab_size
        self.out_hand_vocab_size = out_hand_vocab_size
        # TODO: Move to this
        # self.out_vocab_sizes = {
        #     "offset": out_offset_vocab_size,
        #     "downbeat": out_downbeat_vocab_size,
        #     "duration": out_duration_vocab_size,
        #     "pitch": out_pitch_vocab_size,
        #     "accidental": out_accidental_vocab_size,
        #     "keysignature": out_keysignature_vocab_size,
        #     "velocity": out_velocity_vocab_size,
        #     "grace": out_grace_vocab_size,
        #     "trill": out_trill_vocab_size,
        #     "staccato": out_staccato_vocab_size,
        #     "voice": out_voice_vocab_size,
        #     "stem": out_stem_vocab_size,
        #     "hand": out_hand_vocab_size,
        # }
        self.is_autoregressive = is_autoregressive
        assert positional_encoding in ["RoPE", "ALiBi", "absolute"]
        self.positional_encoding = positional_encoding
        self.conditional_sampling = conditional_sampling
        self.bias = bias
