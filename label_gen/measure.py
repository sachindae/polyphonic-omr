# Wrapper class for parse tree of measure element

class Measure:

    def __init__(self, measure, num_staves, beats, beat_type):

        # Store measure .xml along with number of staves and time signature info
        self.measure = measure
        self.num_staves = num_staves
        self.beats = beats
        self.beat_type = beat_type


    def parse_attributes(self, attributes):
        
        '''
        Reads through all attributes of a measure 
        (this contains key info, time info, etc.)

        attributes: the parse tree representing the attributes
        '''

        sequence = ''
        skip = 0

        # Iterate through all attributes
        for attribute in attributes:

            if attribute.tag == 'key':
                # Sharps are positive, flats neg
                try:
                    sequence += 'keySignature-' + self.num_sharps_flats_to_key(int(attribute[0].text)) + ' '
                except ValueError:
                    # Really weird key change, just skip the file
                    return 'percussion', skip, self.beats, self.beat_type
                    
            elif attribute.tag == 'time':
                # Top and bottom num
                try:
                    self.beats = int(attribute[0].text)
                    self.beat_type = int(attribute[1].text)
                except ValueError: 
                    # Weird error where attribute 0 text is '2+2+3', just skip the file
                    return 'percussion', skip, self.beats, self.beat_type
                
                #print('NEW TIME:',self.beats,self.beat_type)

                if 'symbol' in attribute.attrib:
                    if attribute.attrib['symbol'] == 'cut':     # Cut time
                        sequence += 'timeSignature-C/' + ' '
                    elif attribute.attrib['symbol'] == 'common': # Common time
                        sequence += 'timeSignature-C' + ' '
                    else:           # Default time
                        sequence += 'timeSignature-' + attribute[0].text + '/' + attribute[1].text + ' '
                else:   # Default time sig
                    sequence += 'timeSignature-' + attribute[0].text + '/' + attribute[1].text + ' '

            elif attribute.tag == 'clef' and ('number' not in attribute.attrib or attribute.attrib['number'] == '1'):
                # Clef and line (add this first)
                sequence = 'clef-' + attribute[0].text + attribute[1].text + ' ' + sequence

            elif attribute.tag == 'measure-style':
                # Look for multi-rest/repeats
                s, skip = self.parse_measure_style(attribute)
                sequence += s

        # Add + symbol between if multiple attributes
        seq_split = sequence.split()
        if len(seq_split) > 1:
            sequence = ' + '.join(seq_split) + ' '
            #print('SEQUENCE:', sequence)

        sequence = [sequence for i in range(self.num_staves)]
        return sequence, skip, self.beats, self.beat_type
    
    def parse_note(self, note):

        '''
        Reads through a note of a measure 
        (this contains staff, voice, articulation, pitch info, etc.)

        note: the parse tree representing the note
        '''

        sequence = ['' for x in range(self.num_staves)]
        cur_rest = False        # for differentiating note vs rest

        # Get staff, voice, dot of note, or is part of chord
        staff, voice, has_dot, is_chord, dur, is_grace, stem_down, articulation = 0, 1, False, False, 0, False, True, ''
        for e in note:
            if e.tag == 'staff':    # Staff number of note
                # Only read notes of first staff, skip otherwise
                staff = int(e.text) - 1
                if (staff > 0):
                    return 'multi-staff', True, voice, dur, is_grace, articulation
            if e.tag == 'voice':    # Voice number of note
                voice = int(e.text)
            if e.tag == 'dot':      # Dot modifier for note duration
                has_dot = True
            if e.tag == 'chord':    # Note is part of a chord
                is_chord = True
            if e.tag == 'duration': # Duration of note
                dur = int(e.text)
            if e.tag == 'grace':    # Note is a gracenote
                is_grace=True
            if e.tag == 'stem':     # Direction of stem (not used)
                stem_down = False if e.text == 'up' else True

        # Check that note is printed, skip if not
        if 'print-object' in note.attrib and note.attrib['print-object'] == 'no':
            return 'forward', True, voice, dur, is_grace, articulation

        # Information about the note's pitch and octave
        pitch = ''
        alter = ''
        octave = ''

        # Check for accidentals beforehand due to the indexing in parse tree
        for elem in note:
            if elem.tag == 'accidental':  
                if elem.text == 'sharp':
                    alter = '#'
                elif elem.text == 'flat':
                    alter = 'b'
                elif elem.text == 'natural':
                    alter = 'N'
                elif elem.text == 'double-sharp':
                    alter = '##'
                elif elem.text == 'flat-flat':
                    alter = 'bb'

        # Iterate through all elements in note obj
        for elem in note:

            # Check for the pitch
            if elem.tag == 'pitch':
                for e in elem:
                    if e.tag == 'step':         # Pitch
                        pitch = e.text
                    elif e.tag == 'alter':      # Not used
                        pass
                    elif e.tag == 'octave':     # Octave number
                        octave = e.text
                # Create 
                sequence[staff] += 'note-' + pitch + alter + octave

            # Check for a rest note
            if elem.tag == 'rest':
                # Check if measure rest or has a type
                if 'measure' in elem.attrib and elem.attrib['measure'] == 'yes':
                    # Convert rest-measure to note depending on time signature
                    sequence[staff] += self.rest_measure_to_note() + ' '
                else:
                    sequence[staff] += 'rest' 
                cur_rest = True

            # Check duration of the note
            elif elem.tag == 'type':
                # Length of note
                dot = '. ' if has_dot else ' '
                duration = 'sixteenth' if elem.text == '16th' else \
                           'thirty_second' if elem.text == '32nd' else \
                           'sixty_fourth' if elem.text == '64th' else \
                           'hundred_twenty_eighth' if elem.text == '128th' else \
                            elem.text
                if cur_rest:
                    sequence[staff] += '-' + duration + dot#'-v' + str(voice) + ' '
                    cur_rest = False
                else:
                    sequence[staff] += '_' + duration + dot#'-v' + str(voice) + ' '

            elif elem.tag == 'chord':   # Unused
                pass

            elif elem.tag == 'notations':   # Unused
                articulation = self.parse_notations(elem, stem_down)


        return sequence, is_chord, voice, dur, is_grace, articulation

    def parse_direction(self, direction):

        '''
        Reads through a direction element of a measure (unused)
        (this contains dynamics information)

        note: the parse tree representing the note
        '''

        sequence = ['' for x in range(self.num_staves)]

        # Get staff of note
        '''
        staff = 0
        for e in direction:
            if e.tag == 'staff':
                staff = int(e.text) - 1
        '''

        # Iterate through all elements in direction obj
        '''
        for elem in direction:

            pass

            if elem.tag == 'direction-type':

                if elem[0].tag == 'dynamics':
                    sequence[staff] += elem[0][0].tag + '-dynamic' + ' '

                if elem[0].tag == 'words' and elem[0].text is not None:
                    sequence[staff] += elem[0].text + '-dynamic' + ' '

            elif elem.tag == 'sound':

                if 'tempo' in elem.attrib:
                    pass # don't show tempo for now
                    #sequence[staff] += elem.attrib['tempo'] + '-tempo' + ' '
        '''

        return sequence

    def parse_notations(self, notation, stem_down):

        '''
        Reads through a notation element of a note (unused)
        (this contains articulation information)

        note: the parse tree representing the notation
        stem_down: indicates direction of stem, can be used for finding
                   location of articulation (above vs below)
        '''

        sequence = ''

        has_fermata = False

        # Iterate through all elements in notation obj
        for n in notation:

            if n.tag == 'tied':

                if n.attrib['type'] == 'start':
                    pass
                    #sequence += 'tie' + ' '

            elif n.tag == 'slur':
                pass
                #sequence += 'slur-' +  n.attrib['type'] + ' '
        
            elif n.tag == 'articulations':
                # Go through all articulations
                for articulation in n:
                    sequence += articulation.tag + ' '

            elif n.tag == 'fermata':
                has_fermata = True

        # Add fermata at end (bottom to top, fermata always at top of articulations)
        if has_fermata:
            sequence += 'fermata '

        return sequence

    def parse_measure_style(self, style):

        '''
        Reads through a style element of a measure
        (this contains information about multirests)

        style: the parse tree representing the style
        '''

        sequence = ''
        skip = 0

        # Iterate through all elements in notation obj
        for s in style:

            if s.tag == 'multiple-rest':
                sequence += 'multirest-' + s.text + ' '
                skip = int(s.text)

        return sequence, skip

    def num_sharps_flats_to_key(self, num):

        """
        Converts num sharps/flats to key

        num: indicates num sharps/flat (> 0 is sharp, < 0 is flat)
        """

        mapping = {7: 'C#M', 6: 'F#M', 5: 'BM', 4: 'EM',
                   3: 'AM', 2: 'DM', 1: 'GM', 0: 'CM',
                   -1: 'FM', -2: 'BbM', -3: 'EbM', -4: 'AbM',
                   -5: 'DbM', -6: 'GbM', -7: 'CbM'}

        return mapping[num]

    def rest_measure_to_note(self):

        """
        Converts rest-measure to coresponding value
        based on time signature
        """

        type_map = {
            '1': 'whole', '2': 'half', '4': 'quarter', '8': 'eighth', '12': 'eighth.', '16': 'sixteenth', 
            '32': 'thirthy_second', '48': 'thirthy_second.',
        }

        #note_type = type_map[str(self.beat_type)]
        return 'rest-whole'