"""Tokenizer for music21 streams and pretty_midi objects."""

import math
from fractions import Fraction
from typing import Dict, List

import pretty_midi
import torch
import torch.nn.functional as F
from music21 import (
    articulations,
    clef,
    converter,
    expressions,
    instrument,
    key,
    meter,
    note,
    stream,
    tempo,
)

from music21.common.numberTools import opFrac
from music21.midi.translate import prepareStreamForMidi
from score_utils import realize_spanners


class Downbeat:
    MEASURE_NUMBER = 0
    OFFSET = 1
    LAST_OFFSET = 2
    MEASURE_LENGTH = 3


db_config = Downbeat.MEASURE_LENGTH

PARAMS = {
    "offset": {"min": 0, "max": 6, "step_size": 1 / 24},
    "duration": {"min": 0, "max": 4, "step_size": 1 / 24},
    "downbeat": {"min": -1 / 24, "max": 6, "step_size": 1 / 24},
}


class MultistreamTokenizer:
    @staticmethod
    def midi_to_list(midi_path: str) -> List[pretty_midi.Note]:
        """Converts a MIDI file to a list of notes.
        Used during preprocessing.

        Parameters
        ----------
        midi_path : str
            Path to the midi file.

        Returns
        -------
        List[pretty_midi.Note]
        """
        midi = pretty_midi.PrettyMIDI(midi_path)
        return sorted(
            [n for ins in midi.instruments for n in ins.notes],
            key=lambda n: (n.start, n.pitch, n.end - n.start),
        )

    @staticmethod
    def parse_midi(midi_path: str) -> Dict[str, torch.Tensor]:
        """Converts a MIDI file to a list of tensors.
        No quantization or bucketing is applied yet.
        Used during preprocessing.

        Parameters
        ----------
        midi_path : str
            Path to the midi file.

        Returns
        -------
        Dict[str, torch.Tensor]
            returns a dict of tensors of shape (n_notes,) with keys
            "onset", "duration", "pitch", "velocity"
        """
        midi_list = MultistreamTokenizer.midi_to_list(midi_path)
        return {
            "onset": torch.FloatTensor([n.start for n in midi_list]),
            "duration": torch.FloatTensor([n.end - n.start for n in midi_list]),
            "pitch": torch.LongTensor([n.pitch for n in midi_list]),
            "velocity": torch.LongTensor([n.velocity for n in midi_list]),
        }

    @staticmethod
    def bucket_midi(midi_streams: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Flexible and fast conversion of raw midi values to token representation.

        Parameters
        ----------
        midi_streams : Dict[str, torch.Tensor]
            Dict of tensors of shape (n_notes, ) with keys "onset", "duration", "pitch",
            "velocity".
            Ideally, these tensors are prepared by `MultiStreamTokenizer.parse_midi`.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dict of tensors of shape (n_notes, n_buckets) with keys "onset", "duration",
            "pitch", "velocity" and a padding tensor of shape (n_notes, ) w/ key "pad".
        """
        # one-hot
        onset_stream = torch.diff(midi_streams["onset"], prepend=torch.Tensor([0.0]))
        onset_stream = torch.log(4 * onset_stream + 1) * 4 / math.log(4 * 8.0 + 1)
        onset_stream = one_hot_bucketing(onset_stream, 0, 4, 200)
        # Squash durations logarithmically such that 0 -> 0 and 16 -> 4
        # fmt: off
        duration_stream = midi_streams["duration"]
        duration_stream = (4 * duration_stream + 1).log() * 4 / math.log(4 * 16.0 + 1)
        duration_stream = one_hot_bucketing(duration_stream, 0, 4, 200)
        # fmt: on
        pitch_stream = one_hot_bucketing(midi_streams["pitch"], 0, 127, 128)
        velocity_stream = one_hot_bucketing(midi_streams["velocity"], 0, 127, 8)
        return {
            "onset": onset_stream.float(),
            "duration": duration_stream.float(),
            "pitch": pitch_stream.float(),
            "velocity": velocity_stream.float(),
            "pad": torch.ones((onset_stream.shape[0],), dtype=torch.long),
        }

    @staticmethod
    def tokenize_midi(midi_path) -> Dict[str, torch.Tensor]:
        """Converts a MIDI file to list of tensors.

        Parameters
        ----------
        midi_path : str
            Path to the midi file.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dict of tensors that represent token streams.
        """
        midi_streams = MultistreamTokenizer.parse_midi(midi_path)
        return MultistreamTokenizer.bucket_midi(midi_streams)

    @staticmethod
    def mxl_to_list(mxl_path: str) -> tuple[List[note.Note], stream.Score]:
        """Converts a music21 stream to a sorted and deduplicated list of notes.

        Parameters
        ----------
            mxl_path : str
                Path to the musicxml file.

        Returns
        -------
            List[music21.note.Note]:
                The list of notes in the music21 stream.
            music21.stream.Score:
                The music21 stream. This is only returned to
                ensure that the stream is not garbage collected.
        """
        mxl = converter.parse(mxl_path, forceSource=True)
        mxl = realize_spanners(mxl)
        mxl: stream.Score = mxl.expandRepeats()
        # strip all ties inPlace
        mxl.stripTies(preserveVoices=False, inPlace=True)
        # Realize Tremolos
        for n in mxl.recurse().notes:
            for e in n.expressions:
                if isinstance(e, expressions.Tremolo):
                    offset = n.offset
                    out = e.realize(n, inPlace=True)[0]
                    v = n.activeSite
                    v.remove(n)
                    for n2 in out:
                        v.insert(offset, n2)
                        offset += n2.duration.quarterLength
                    break
        mxl = prepareStreamForMidi(mxl)

        notes: list[note.Note] = []
        assert not any(note.isChord for note in mxl.flatten().notes)

        for n in mxl.flatten().notes:
            # if note.style.noteSize == "cue":
            #     continue
            if n.style.hideObjectOnPrint:
                continue
            n.volume.velocity = int(round(n.volume.cachedRealized * 127))
            notes.append(n)
        # Sort like this to preserve correct order for grace notes.
        def sortTuple(n):
           # Sort by offset, then pitch, then duration
           # Grace notes that share the same offset are sorted by their insertIndex
           # instead of their pitch as they rarely actually occur simultaneously
           return (
               n.offset,
               not n.duration.isGrace,
               n.pitch.midi if not n.duration.isGrace else n.sortTuple(mxl).insertIndex,
               n.duration.quarterLength
           )
        #    return (n.offset, n.pitch.midi, n.duration.quarterLength)
        notes_sorted = sorted(notes, key=sortTuple)
        notes_consolidated: list[note.Note] = []
        last_note = None
        for n in notes_sorted:
            if last_note is None or n.offset != last_note.offset or n.pitch.midi != last_note.pitch.midi:
                notes_consolidated.append(n)
                last_note = n
            elif last_note.duration.isGrace:
                last_note = n
            else:
                if n.duration.quarterLength > last_note.duration.quarterLength:
                    last_note = n
        # sort again because we might have changed the duration of grace notes
        notes_consolidated = sorted(notes_consolidated, key=sortTuple)
        return notes_consolidated, mxl

    @staticmethod
    def parse_mxl(mxl_path) -> Dict[str, torch.Tensor]:
        """
        Converts a MusixXML file to a list of tensors.
        All tensors have shape (n_notes,) and no quantization is applied yet.
        Used during preprocessing.

        Parameters
        ----------
        mxl_path : str
            Path to the musicxml file.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dict of tensors of shape (n_notes, 1) with keys "offset"
            "downbeat", "duration", "pitch", "accidental", "velocity", "grace", "trill",
            "staccato", "voice", "stem", "hand".
        """
        # return mxl_stream for garbage collection reasons only
        mxl_list, mxl_stream = MultistreamTokenizer.mxl_to_list(mxl_path)
        if len(mxl_list) == 0:
            offset_stream = torch.Tensor([])
            downbeat_stream = torch.Tensor([])
            duration_stream = torch.Tensor([])
            pitch_stream = torch.Tensor([])
            accidental_stream = torch.Tensor([])
            keysignature_stream = torch.Tensor([])
            velocity_stream = torch.Tensor([])
            grace_stream = torch.Tensor([])
            trill_stream = torch.Tensor([])
            staccato_stream = torch.Tensor([])
            voice_stream = torch.Tensor([])
            stem_stream = torch.Tensor([])
            hand_stream = torch.Tensor([])
        else:
            # fmt: off
            note_offsets = torch.FloatTensor([n.offset for n in mxl_list])
            measure_offsets = torch.FloatTensor([n.getContextByClass("Measure").offset for n in mxl_list])
            offset_stream = note_offsets - measure_offsets

            if db_config == Downbeat.MEASURE_NUMBER:
                nums = torch.tensor([n.getContextByClass("Measure").number for n in mxl_list])
                downbeat_stream = (torch.diff(nums, prepend=torch.tensor([1])) > 0).float()
            elif db_config == Downbeat.OFFSET:
                downbeat_stream = torch.logical_or(offset_stream == 0, torch.diff(offset_stream, prepend=torch.tensor([0.0])) < 0).float()
            elif db_config == Downbeat.LAST_OFFSET:
                downbeat_stream = torch.diff(measure_offsets, prepend=torch.tensor([0.0])) > 0
                shifts = measure_offsets - torch.cat((torch.tensor([0]), note_offsets[:-1]))
                downbeat_stream = torch.where(downbeat_stream, shifts, torch.ones_like(downbeat_stream).float() * PARAMS["downbeat"]["min"])
            elif db_config == Downbeat.MEASURE_LENGTH:
                downbeat_stream = torch.diff(measure_offsets, prepend=torch.tensor([0.0]))
                downbeat_stream[downbeat_stream<=0] = PARAMS["downbeat"]["min"]

            duration_stream = torch.Tensor([n.duration.quarterLength for n in mxl_list])
            pitch_stream = torch.Tensor([n.pitch.midi for n in mxl_list])
            velocity_stream = torch.Tensor([n.volume.velocity for n in mxl_list])
            def alter_map(accidental):
                if accidental is None:
                    return 5
                alter_to_value = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
                # if not in the mapping, return 6 (for unknown)
                return alter_to_value.get(accidental.alter, 6)
            accidental_stream = torch.Tensor([alter_map(n.pitch.accidental) for n in mxl_list])
            # for each note offset, find the last key that occurs before or at the same time as it
            keysignatures = {float(e.offset): e for e in mxl_stream.flatten().getElementsByClass(key.KeySignature)}
            keysignature_stream = torch.Tensor([next(((v.sharps if v.sharps is not None else 8) for k, v in reversed(keysignatures.items()) if k <= n), 8) for n in note_offsets]) + 7
            # MusicXML attribute streams
            grace_stream = torch.Tensor([n.duration.isGrace for n in mxl_list])
            trills = (expressions.Trill, expressions.InvertedMordent, expressions.Mordent, expressions.Turn)
            trill_stream = torch.Tensor([any(isinstance(e, trills) for e in n.expressions) for n in mxl_list])
            staccatos = (articulations.Staccatissimo, articulations.Staccato)
            staccato_stream = torch.Tensor([any(isinstance(e, staccatos) for e in n.articulations) for n in mxl_list])
            voices = [n.getContextByClass("Voice") for n in mxl_list]
            voice_stream = torch.Tensor([int(v.id) if v is not None else 0 for v in voices])
            stem_map = {"up": 0, "down": 1, "noStem": 2}
            stem_stream = torch.Tensor([stem_map.get(n.stemDirection, 3) for n in mxl_list])
            # fmt: on
            # Hands/Staff logic is slightly more complicated
            #
            hand_stream = []
            not_matched = set()
            for n in mxl_list:
                # Usually part names are similar to "[P1-Staff2]"
                part_name = n.getContextByClass("Part").id.lower()
                if "staff1" in part_name:
                    hand_stream.append(0)
                elif "staff2" in part_name:
                    hand_stream.append(1)
                else:
                    hand_stream.append(2)
                    if part_name not in not_matched:  # only one warning per part
                        not_matched.add(part_name)
                        # print("Couldn't match", part_name)
            hand_stream = torch.tensor(hand_stream)
        mxl_stream  # keep stream for gc only
        return {
            "offset": offset_stream,
            "downbeat": downbeat_stream,
            "duration": duration_stream,
            "pitch": pitch_stream,
            "accidental": accidental_stream,
            "keysignature": keysignature_stream,
            "velocity": velocity_stream,
            "grace": grace_stream,
            "trill": trill_stream,
            "staccato": staccato_stream,
            "voice": voice_stream,
            "stem": stem_stream,
            "hand": hand_stream,
        }

    @staticmethod
    def bucket_mxl(mxl_streams: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Bucketing TODO: checkout bucketing
        # fmt: off
        offset_stream = one_hot_bucketing(mxl_streams["offset"], **PARAMS["offset"])
        duration_stream = one_hot_bucketing(mxl_streams["duration"], **PARAMS["duration"])
        downbeat_stream = one_hot_bucketing(mxl_streams["downbeat"], **PARAMS["downbeat"])
        pitch_stream = one_hot_bucketing(mxl_streams["pitch"], 0, 127, 128)
        accidental_stream = one_hot_bucketing(mxl_streams["accidental"], 0, 6, 7)
        keysignature_stream = one_hot_bucketing(mxl_streams["keysignature"], 0, 15, 16)
        velocity_stream = one_hot_bucketing(mxl_streams["velocity"], 0, 127, 8)
        grace_stream = one_hot_bucketing(mxl_streams["grace"], 0, 1, 2)
        trill_stream = one_hot_bucketing(mxl_streams["trill"], 0, 1, 2)
        staccato_stream = one_hot_bucketing(mxl_streams["staccato"], 0, 1, 2)
        voice_stream = one_hot_bucketing(mxl_streams["voice"], 0, 8, 9)
        stem_stream = one_hot_bucketing(mxl_streams["stem"], 0, 3, 4)
        hand_stream = one_hot_bucketing(mxl_streams["hand"], 0, 2, 3)
        # fmt: on
        # Beams
        # Slurs
        # Tuplets
        # Dots?
        return {
            "offset": offset_stream.float(),
            "downbeat": downbeat_stream.float(),
            "duration": duration_stream.float(),
            "pitch": pitch_stream.float(),
            "accidental": accidental_stream.float(),
            "keysignature": keysignature_stream.float(),
            "velocity": velocity_stream.float(),
            "grace": grace_stream.float(),
            "trill": trill_stream.float(),
            "staccato": staccato_stream.float(),
            "voice": voice_stream.float(),
            "stem": stem_stream.float(),
            "hand": hand_stream.float(),
            "pad": torch.ones((offset_stream.shape[0],), dtype=torch.long),
        }

    @staticmethod
    def tokenize_mxl(mxl_path: str) -> Dict[str, torch.Tensor]:
        """Converts a MusicXML file to a list of tensors of shape (n_notes).

        Parameters
        ----------
        mxl_path : str
            Path to the musicxml file.

        Returns
        -------
        torch.Tensor
            returns a list of tensors of shape (n_notes,)
        """
        mxl_streams = MultistreamTokenizer.parse_mxl(mxl_path)
        return MultistreamTokenizer.bucket_mxl(mxl_streams)

    @staticmethod
    def detokenize_mxl(token_dict: Dict[str, torch.Tensor], midi_sequence: List[pretty_midi.Note]|None= None) -> stream.Score:
        """Decode the token streams into a music21 stream that can be saved to musicxml.
        The surprising complexity comes from incompatibilities in music21's XML export.
        This function is tested such that saving and reloading the musicxml file should
        yield the same score. This is not the case for music21's makeNotation function.

        Parameters
        ----------
        token_dict : Dict[str, torch.Tensor]
            Dict of tensors of shape (n_notes, n_buckets) with keys "offset", "duration",
            "pitch", "velocity" and a padding tensor of shape (n_notes, ) with key "pad".

        Returns
        -------
        music21.stream.Stream
            music21 stream that can be saved to musicxml.
        """
        mask = token_dict["pad"].squeeze() > 0.5  # allow for prediction/soft values
        # fmt: off
        offset_stream = one_hot_unbucketing(token_dict["offset"][mask], **PARAMS["offset"]).numpy().astype(float)
        duration_stream = one_hot_unbucketing(token_dict["duration"][mask], **PARAMS["duration"]).numpy().astype(float)
        downbeat_stream = one_hot_unbucketing(token_dict["downbeat"][mask], **PARAMS["downbeat"]).numpy().astype(float)
        pitch_stream = one_hot_unbucketing(token_dict["pitch"][mask], 0, 127, 128).numpy().astype(int)
        accidental_stream = one_hot_unbucketing(token_dict["accidental"][mask][:, :6], 0, 6, 7).numpy().astype(int)
        keysignature_stream = one_hot_unbucketing(token_dict["keysignature"][mask], 0, 15, 16).numpy().astype(int)
        velocity_stream = one_hot_unbucketing(token_dict["velocity"][mask], 0, 127, 8).numpy().astype(int)
        grace_stream = one_hot_unbucketing(token_dict["grace"][mask], 0, 1, 2).numpy().astype(bool)
        trill_stream = one_hot_unbucketing(token_dict["trill"][mask], 0, 1, 2).numpy().astype(bool)
        staccato_stream = one_hot_unbucketing(token_dict["staccato"][mask], 0, 1, 2).numpy().astype(bool)
        voice_stream = one_hot_unbucketing(token_dict["voice"][mask][:, 1:], 1, 8, 8).numpy().astype(int)
        stem_stream = one_hot_unbucketing(token_dict["stem"][mask][:, :3], 0, 3, 4).numpy().astype(int)
        hand_stream = one_hot_unbucketing(token_dict["hand"][mask][:, :2], 0, 2, 3).numpy().astype(int)

        if midi_sequence is not None:
            midi_sequence = [m for i, m in enumerate(midi_sequence) if mask[i]]
        # fmt: on
        measures: list[list[stream.Measure]] = [[], []]
        active_voices_list: list[list[set[int]]] = [[], []]
        # We go through all notes twice, once for each hand.
        # We create measures/increment times etc. both times.
        # However, only the notes for the current part are inserted.
        # This is highly inefficient, but ensures correctness for now.
        for part in range(2):
            active_voices = set()
            m = stream.Measure(number=1)
            voices = [stream.Voice(id=str(i)) for i in range(1, 17)]
            previous_note_is_downbeat = True
            last_measure_duration = None
            last_keysignature = None
            for i in range(len(offset_stream)):
                if db_config == Downbeat.MEASURE_NUMBER:
                    if downbeat_stream[i]:
                        for v in voices:
                            m.insert(0, v)
                        measures[part].append(m)
                        active_voices_list[part].append(active_voices)
                        active_voices = set()
                        voices = [stream.Voice(id=str(i)) for i in range(1, 17)]
                        m = stream.Measure(m.number + 1)
                elif db_config == Downbeat.OFFSET:
                    if (
                        downbeat_stream[i] == 1
                        and offset_stream[i] <= offset_stream[i - 1]
                    ) or (i > 0 and offset_stream[i] < offset_stream[i - 1]):
                        if not previous_note_is_downbeat:
                            for v in voices:
                                m.insert(0, v)
                            measures[part].append(m)
                            active_voices_list[part].append(active_voices)
                            active_voices = set()
                            voices = [stream.Voice(id=str(i)) for i in range(1, 17)]
                            m = stream.Measure(m.number + 1)
                        previous_note_is_downbeat = True
                    else:
                        previous_note_is_downbeat = False
                elif db_config in (Downbeat.LAST_OFFSET, Downbeat.MEASURE_LENGTH):
                    if (
                        1 < i + 1 < len(downbeat_stream)
                        and downbeat_stream[i] >= 0
                        and not (
                            downbeat_stream[i + 1] >= 0
                            and offset_stream[i - 1] != 0
                            and offset_stream[i + 1] <= offset_stream[i]
                        )
                    ):
                        if midi_sequence is not None and i < len(midi_sequence):
                            # If we have the input midi timings, we can use them to set the tempo
                            # We first set tempo marks to track where their location `should` be
                            # The inserted tempo marks therefore form (offset, time in seconds) pairs.
                            for s in range(3):
                                if midi_sequence[min(i+s, len(midi_sequence))].pitch == pitch_stream[i]:
                                    m.insert(opFrac(offset_stream[i]), tempo.MetronomeMark(number=midi_sequence[min(i+s, len(midi_sequence))].start))
                                    break
                                elif midi_sequence[max(i-s, 0)].pitch == pitch_stream[i]:
                                    m.insert(opFrac(offset_stream[i]), tempo.MetronomeMark(number=midi_sequence[max(i-s, 0)].start))
                                    break
                            else:
                                m.insert(opFrac(offset_stream[i]), tempo.MetronomeMark(number=midi_sequence[i].start))
                        for v in voices:
                            m.insert(0, v)
                        if db_config == Downbeat.MEASURE_LENGTH:
                            duration = opFrac(downbeat_stream[i])
                        else:
                            duration = opFrac(offset_stream[i - 1] + downbeat_stream[i])

                        def find_time_signature(measure_length: float) -> meter.TimeSignature|None:
                            frac = Fraction(measure_length / 4 + 1e-5).limit_denominator(16)
                            if frac.numerator != 0:
                                if frac.denominator == 1:
                                    return meter.TimeSignature(f"{frac.numerator * 4}/4")
                                elif frac.denominator == 2:
                                    return meter.TimeSignature(f"{frac.numerator * 4}/8")
                                elif frac.denominator == 4:
                                    return meter.TimeSignature(f"{frac.numerator}/4")
                                elif frac.denominator == 8:
                                    return meter.TimeSignature(f"{frac.numerator}/8")
                                elif frac.denominator == 16:
                                    return meter.TimeSignature(f"{frac.numerator}/16")
                            return None

                        if duration != 0 and duration != last_measure_duration:
                            ts = find_time_signature(duration)
                            if ts is not None:
                                m.insert(0, ts)
                            elif last_measure_duration is None:  # first measure
                                m.insert(0, meter.TimeSignature("4/4"))
                            last_measure_duration = duration
                        measures[part].append(m)
                        active_voices_list[part].append(active_voices)
                        active_voices = set()
                        voices = [stream.Voice(id=str(i)) for i in range(1, 17)]
                        m = stream.Measure(m.number + 1)
                if keysignature_stream[i] != last_keysignature:
                    m.insert(0, key.KeySignature(keysignature_stream[i] - 7))
                    last_keysignature = keysignature_stream[i]
                # inefficient but don't care for now
                if hand_stream[i] == part:
                    n = note.Note()
                    n.duration.quarterLength = opFrac(duration_stream[i])
                    # Adding accidentals shifts the pitch step so we have to account for that
                    # by offsetting the other way first
                    accidental_mapping = {
                        0: (+2, "double-flat"),
                        1: (+1, "flat"),
                        2: (0, "natural"),
                        3: (-1, "sharp"),
                        4: (-2, "double-sharp")
                    }
                    midi_adjustment, accidental_name = accidental_mapping.get(accidental_stream[i], (0, None))
                    n.pitch.midi = pitch_stream[i] + midi_adjustment
                    if accidental_name is not None:
                        n.pitch.accidental = accidental_name
                    else:
                        # Handle the case where the accidental_stream value is outside the expected range
                        n.pitch.midi = pitch_stream[i]
                    if n.pitch.midi != pitch_stream[i]:
                        print(f"Mismatch: {n.pitch.midi} != {pitch_stream[i]}")
                        n.pitch.midi = pitch_stream[i]

                    # n.volume.velocity = velocity_stream[i]
                    if trill_stream[i]:
                        n.expressions.append(expressions.Trill())
                    if staccato_stream[i]:
                        n.articulations.append(articulations.Staccato())
                    if grace_stream[i] or n.duration.quarterLength == 0:
                        # obscure bug in makeNotation forces us to set the duration to 0
                        n.duration.quarterLength = 0
                        n.duration = n.duration.getGraceDuration()
                    stem_map = {0: "up", 1: "down", 2: "noStem"}
                    n.stemDirection = stem_map[stem_stream[i]]

                    v = voice_stream[i]
                    # We need to find a voice that is not active at the current offset.
                    # (also have to consider the previous part!).
                    def find_suitable_voice(v):
                        candidates = sorted(
                            range(len(voices)), key=lambda x: (abs(x - v), -x)
                        )
                        not_ideal = None
                        for candidate in candidates:
                            # Voice already used in other part?
                            if part == 1 and len(measures[1]) < len(measures[0]):
                                if candidate in active_voices_list[0][len(measures[1])]:
                                    continue
                            # Voice already used in current measure at current timestep?
                            if candidate in active_voices:
                                o = opFrac(offset_stream[i])
                                for n in voices[candidate].notes:
                                    if opFrac(n.offset + n.duration.quarterLength) > o:
                                        break
                                else:
                                    return candidate
                            elif not_ideal is None:
                                not_ideal = candidate
                        return v if not_ideal is None else not_ideal

                    v_new = find_suitable_voice(v - 1)
                    active_voices.add(v_new)
                    voices[v_new].insert(opFrac(offset_stream[i]), n)
            if midi_sequence is not None and i < len(midi_sequence):
                for s in range(3):
                    if midi_sequence[max(i-s, 0)].pitch == pitch_stream[i]:
                        m.insert(opFrac(offset_stream[i]), tempo.MetronomeMark(number=midi_sequence[max(i-s, 0)].start))
                        break
                else:
                    m.insert(opFrac(offset_stream[i]), tempo.MetronomeMark(number=midi_sequence[i].start))
            for v in voices:
                m.insert(0, v)
            measures[part].append(m)
            active_voices_list[part].append(active_voices)

        s = stream.Score()
        if db_config not in (Downbeat.LAST_OFFSET, Downbeat.MEASURE_LENGTH):
            lastDuration = -1
            for m0, m1 in zip(measures[0], measures[1]):
                try:
                    if m0.flatten().highestTime >= m1.flatten().highestTime:
                        ts = m0.bestTimeSignature()
                    else:
                        ts = m1.bestTimeSignature()
                except meter.MeterException:
                    print(f"Couldn't find time signature : ({m0.highestTime} {m1.highestTime}, {m0.duration.quarterLength}, {m1.duration.quarterLength})")
                    continue
                if ts.barDuration.quarterLength != lastDuration:
                    m0.timeSignature = ts
                    m1.timeSignature = ts
        for part in range(2):
            p = stream.Part()
            # Inserting an instrument is required to ensure deterministic part names
            # MUSTER does not play nice with names other than P1, P2, ..., so we force that here.
            ins = instrument.Instrument()
            ins.partId = f"P{part+1}"
            p.insert(0, ins)
            offset = 0
            for i, m in enumerate(measures[part]):
                # Special case pickup measure
                if i == 0:
                    c = clef.TrebleClef() if part == 0 else clef.BassClef()
                    m.insert(0, c)
                    p.insert(0, m)
                else:
                    p.insert(offset, m)
                offset += m.barDuration.quarterLength
            s.insert(0, p)
        return s


def one_hot_bucketing(
    values: torch.Tensor | List[int | float], min, max, buckets=None, step_size=None
) -> torch.Tensor:
    assert buckets is not None or step_size is not None
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    if values.ndim == 2:
        values = values.squeeze(1)
    values = values.float()

    # discretize the values into buckets
    if buckets is None:
        buckets = int((max + step_size - min) / step_size)
        bucket_indices = ((values - min) / (max + step_size - min) * buckets).round()
    else:
        bucket_indices = (values - min) / (max - min) * buckets
    # clamp the bucket indices to be between 0 and n_buckets - 1
    bucket_indices = bucket_indices.long().clamp(0, buckets - 1)
    one_hots = F.one_hot(bucket_indices, num_classes=buckets)
    return one_hots


def one_hot_unbucketing(
    one_hots: torch.Tensor | List[int | float], min, max, buckets=None, step_size=None
) -> torch.FloatTensor:
    assert buckets is not None or step_size is not None
    if not isinstance(one_hots, torch.Tensor):
        one_hots = torch.tensor(one_hots)

    # Convert the one-hot vectors back into bucket indices
    bucket_indices = torch.argmax(one_hots, dim=-1)
    # Convert the bucket indices back into the original values
    if step_size is None:
        step_size = (max + 1 - min) / buckets
    values = min + bucket_indices.float() * step_size
    return values


def positional_embedding(times, dim=128) -> torch.Tensor:
    if not isinstance(times, torch.Tensor):
        times = torch.tensor(times)  # (T, D)
    divisors = torch.tensor([10000 ** (2 * (i // 2) / dim) for i in range(dim)])  # D
    position_enc = times.unsqueeze(-1) / divisors.unsqueeze(0)
    sines = torch.sin(position_enc[:, 0::2])
    cosines = torch.cos(position_enc[:, 1::2])
    return torch.cat([sines, cosines], dim=-1)


if __name__ == "__main__":
    import argparse
    import sys

    from score_transformer import score_similarity

    sys.path.append("../")

    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", type=str, default=None)
    parser.add_argument(
        "--mxl",
        type=str,
        default="./data/asap-dataset/Bach/Fugue/bwv_846/xml_score.musicxml",
    )
    args = parser.parse_args()

    if args.midi is not None:
        tokenized = MultistreamTokenizer.tokenize_midi(args.midi)
        print(tokenized)

    if args.mxl is not None:
        mxl, _ = MultistreamTokenizer.mxl_to_list(args.mxl)
        tokenized = MultistreamTokenizer.tokenize_mxl(args.mxl)
        s_recon = MultistreamTokenizer.detokenize_mxl(tokenized)
        s = converter.parse(args.mxl, forceSource=True).expandRepeats()
        print(score_similarity(s_recon, s))
