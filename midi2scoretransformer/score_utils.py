
import copy
import os
import shutil
import subprocess
import tempfile

from music21 import chord, expressions, key, note, stream

from constants import LD_PATH, MUSESCORE_PATH


def realize_spanners(s):
    to_remove = []
    for sp in s.recurse().getElementsByClass(expressions.TremoloSpanner):
        l = sp.getSpannedElements()
        if len(l) != 2:
            print("Not sure what to do with this spanner", sp, l)
            continue
        start, end = l

        offset = start.offset
        start_chord = None
        end_chord = None
        startActiveSite = start.activeSite
        endActiveSite = end.activeSite
        if start.activeSite is None:
            start_chord: chord.Chord = start._chordAttached
            offset = start_chord.offset
            startActiveSite = start._chordAttached.activeSite
        if end.activeSite is None:
            end_chord: chord.Chord = end._chordAttached
            endActiveSite = end._chordAttached.activeSite

        # We insert a tremolo expression on the start note
        # realize it, and then change every second note to have the pitch of the end note
        trem = expressions.Tremolo()
        trem.measured = sp.measured
        trem.numberOfMarks = sp.numberOfMarks
        start.expressions.append(trem)
        out = trem.realize(start, inPlace=True)[0]
        if start_chord:
            if len(start_chord.notes) == 1:
                startActiveSite.remove(start_chord)
            else:
                start_chord.remove(start)
        else:
            startActiveSite.remove(start)
        if end_chord:
            if len(end_chord.notes) == 1:
                endActiveSite.remove(end_chord)
            else:
                end_chord.remove(end)
        else:
            endActiveSite.remove(end)
        for i, n2 in enumerate(out):
            if i % 2 == 1:
                n2.pitch = end.pitch
            startActiveSite.insert(offset, n2)
            offset += n2.duration.quarterLength
        to_remove.append(sp)
    for sp in s.recurse().getElementsByClass(expressions.TrillExtension):
        l = sp.getSpannedElements()
        start = l[0]
        exp = [l.expressions for l in l]
        if not any(isinstance(e, expressions.Trill) for ex in exp for e in ex):
            if len(l) != 1:
                print("Not sure what to do with this spanner", sp, l)
                continue
            start.expressions.append(expressions.Trill())
            to_remove.append(sp)
    s.remove(to_remove, recurse=True)
    return s


def convert_with_musescore(in_path: str, out_path: str):
    with tempfile.TemporaryDirectory() as tmpdirname:
        suffix = in_path.split(".")[-1]
        shutil.copy(in_path, f"{tmpdirname}/test.{suffix}")
        # Update the environment variables
        env_vars = os.environ.copy()
        env_vars["LD_LIBRARY_PATH"] = LD_PATH + env_vars.get("LD_LIBRARY_PATH", "")
        env_vars["DISPLAY"] = ":0"
        env_vars["QT_QPA_PLATFORM"] = "offscreen"
        env_vars["XDG_RUNTIME_DIR"] = tmpdirname
        # Run the subprocess with the updated environment
        subprocess.run(
            [MUSESCORE_PATH, "-o", out_path, f"{tmpdirname}/test.{suffix}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env_vars
        )




def postprocess_score(mxl: stream.Score, makeChords: bool=False, inPlace=False) -> stream.Score:
    """We essentially roll our own "makeNotation" here because music21's is broken
    for non-ideal scores.

    Parameters
    ----------
    mxl : stream.Score
        The score to postprocess.
    makeChords : bool, optional
        Whether to merge notes into chords and do various other prettifications,
        by default False, since it can alter the metric results.

    Returns
    -------
    stream.Score
        The postprocessed score.
    """
    if not inPlace:
        mxl = copy.deepcopy(mxl)

    def remove_note_fast(n: note.Note) -> None:
        """Quickly removes a note known to be in a voice.
        Faster as it doesn't require recursion.
        """
        for site in n.sites:
            if isinstance(site, stream.Voice):
                site.remove(n)
                break
        else:
            v = n.getContextByClass(stream.Voice)
            # getContextByClass silently returns incorrect results sometimes
            if n not in v:
                raise ValueError("Note not found in voice")
            v.remove(n)

    # see if any measures can be merged
    merge_candidates: set[tuple[int, int]] = set()
    not_candidates: set[tuple[int, int]] = set()
    for part in mxl.parts:
        measures: list[stream.Measure] = list(part.getElementsByClass("Measure"))
        for i, (this_m, next_m) in enumerate(zip(measures, measures[1:])):
            # see if we can merge two adjacent measures into one
            this_m_highest_offset = max([n.offset for n in this_m.flatten().notes if n.offset < this_m.barDuration.quarterLength], default=0)
            next_m_lowest_offset = min([n.offset for n in next_m.flatten().notes], default=float("inf"))
            if this_m_highest_offset <= next_m_lowest_offset and next_m_lowest_offset != 0 and this_m.barDuration.quarterLength <= next_m.barDuration.quarterLength:
                if len(next_m.flatten().notes) > 6:
                    not_candidates.add((i, i+1))
                    continue
                merge_candidates.add((i, i+1))
            else:
                not_candidates.add((i, i+1))

    remove = sorted(list(merge_candidates-not_candidates), reverse=True)
    for i, j in remove:
        for part in mxl.parts:
            measures: list[stream.Measure] = list(part.getElementsByClass("Measure"))
            this_m = measures[i]
            next_m = measures[j]
            if this_m.barDuration.quarterLength < next_m.barDuration.quarterLength:
                if (ts := this_m.getElementsByClass("TimeSignature")):
                    this_m.remove(ts[0])
                this_m.insert(0, next_m.getElementsByClass("TimeSignature")[0])
            for next_v in next_m.voices:
                this_m.insert(next_v.offset, copy.deepcopy(next_v))
            shift = part.elementOffset(next_m) - part.elementOffset(this_m)
            for m in measures[j:]:
                m.number -= 1
                part.coreSetElementOffset(m, part.elementOffset(m) - shift)
            part.coreElementsChanged(clearIsSorted=False)
            part.remove(next_m)
    # remove doubled notes:
    mxl = mxl.splitAtDurations(recurse=True)[0]
    mxl.makeTies(inPlace=True)
    notes: dict[tuple[float,int,bool], list[note.Note]] = {}
    for n in mxl.flatten().notes:
        key_tuple = (n.offset, n.pitch.midi, n.duration.isGrace)
        notes.setdefault(key_tuple, []).append(n)
    for v in notes.values():
        if len(v) > 1:
            longest_note = max(v, key=lambda x: x.duration.quarterLength)
            for n in v:
                if n is not longest_note:
                    remove_note_fast(n)
    # Hide unnecessary accidentals
    for p in mxl.parts:
        prior_accidental = {i: None for i in range(128)}
        for n in p.flatten():
            if isinstance(n, key.KeySignature):
                kind = "sharp" if n.sharps > 0 else "flat"
                steps = [p.midi % 12 for p in n.alteredPitches]
                for i in range(128):
                    if i % 12 in steps:
                        prior_accidental[i] = kind
            elif isinstance(n, note.Note):
                if n.pitch.accidental is not None:
                    n.pitch.accidental.displayStatus = True
                    if n.pitch.accidental.name == prior_accidental[n.pitch.midi]:
                        n.pitch.accidental.displayStatus = False
    mxl.streamStatus.accidentals = True
    for part in mxl.parts:
        flattened_notes = list(part.flatten().notes)
        offset_duration_dict: dict[tuple,list[note.Note]] = {}
        for n in flattened_notes:
            key_tuple = (n.offset, n.duration.quarterLength, n.duration.isGrace)
            offset_duration_dict.setdefault(key_tuple, []).append(n)
        for notes in offset_duration_dict.values():
            if len(notes) > 1: # Merge needed
                c = chord.Chord(notes)
                first_note: note.Note = notes[0]
                c.expressions = first_note.expressions
                c.articulations = first_note.articulations

                # This is faster than part.replace(notes[0], c, recurse=True):
                for site in first_note.sites:
                    if isinstance(site, stream.Voice):
                        v = site
                        break
                else:
                    v = first_note.getContextByClass(stream.Voice)
                off = v.elementOffset(first_note)
                v.remove(first_note)
                v.insert(off, c)
                c.activeSite = v
                # Remove notes that were merged into the chord
                # Removing one-by-one is faster than removing all at once because we
                # know the direct site of the note, so can avoid recursion.
                for n in notes[1:]:
                    remove_note_fast(n)
    # The following code is fairly ugly cleanup to ensure proper MusicXML export.
    # It removes empty voices and pads under-full measures with rests.
    def merge_and_pad_voices(m):
        voices = list(m.voices)
        non_empty_voices = [voice for voice in voices if len(voice.notes) > 0]
        if non_empty_voices:  # we can remove all voices that only contain a rest
            for v in voices:
                if v not in non_empty_voices:
                    m.remove(v)
        else:  # we just keep a single voice with a full duration rest
            m.remove(voices)
            v = stream.Voice()
            v.id = "1"
            rest = note.Rest(quarterLength=m.barDuration.quarterLength)
            v.append(rest.splitAtDurations())
            m.insert(0, v)
            return
        # pad non-full voices with rests
        for source, v in enumerate(m.voices):
            v: stream.Voice
            # Clean up overlaps
            v.id = str(source + 1)
            if m.highestTime < m.barDuration.quarterLength:
                quarterLength = m.barDuration.quarterLength - v.highestTime
                rest = note.Rest(quarterLength=quarterLength)
                v.append(rest.splitAtDurations())


    for part in mxl.parts:
        measures = list(part.getElementsByClass("Measure"))
        for m in measures:
            merge_and_pad_voices(m)

    mxl = mxl.splitAtDurations(recurse=True)[0]
    return mxl

