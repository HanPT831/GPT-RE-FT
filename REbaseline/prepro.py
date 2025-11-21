from tqdm import tqdm
import ujson as json


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


class Processor:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.new_tokens = []
        if self.args.input_format == 'entity_marker':
            self.new_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']
        self.tokenizer.add_tokens(self.new_tokens)
        if self.args.input_format not in ('entity_mask', 'entity_marker', 'entity_marker_punct', 'typed_entity_marker', 'typed_entity_marker_punct'):
            raise Exception("Invalid input format!")

    def tokenize(self, tokens, subj_type, obj_type, ss, se, os, oe):
        """
        Implement the following input formats:
            - entity_mask: [SUBJ-NER], [OBJ-NER].
            - entity_marker: [E1] subject [/E1], [E2] object [/E2].
            - entity_marker_punct: @ subject @, # object #.
            - typed_entity_marker: [SUBJ-NER] subject [/SUBJ-NER], [OBJ-NER] obj [/OBJ-NER]
            - typed_entity_marker_punct: @ * subject ner type * subject @, # ^ object ner type ^ object #
        """
        sents = []
        input_format = self.args.input_format
        if input_format == 'entity_mask':
            subj_type = '[SUBJ-{}]'.format(subj_type)
            obj_type = '[OBJ-{}]'.format(obj_type)
            for token in (subj_type, obj_type):
                if token not in self.new_tokens:
                    self.new_tokens.append(token)
                    self.tokenizer.add_tokens([token])
        elif input_format == 'typed_entity_marker':
            subj_start = '[SUBJ-{}]'.format(subj_type)
            subj_end = '[/SUBJ-{}]'.format(subj_type)
            obj_start = '[OBJ-{}]'.format(obj_type)
            obj_end = '[/OBJ-{}]'.format(obj_type)
            for token in (subj_start, subj_end, obj_start, obj_end):
                if token not in self.new_tokens:
                    self.new_tokens.append(token)
                    self.tokenizer.add_tokens([token])
        elif input_format == 'typed_entity_marker_punct':
            subj_type = self.tokenizer.tokenize(subj_type.replace("_", " ").lower())
            obj_type = self.tokenizer.tokenize(obj_type.replace("_", " ").lower())

        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)

            if input_format == 'entity_mask':
                if ss <= i_t <= se or os <= i_t <= oe:
                    tokens_wordpiece = []
                    if i_t == ss:
                        new_ss = len(sents)
                        tokens_wordpiece = [subj_type]
                    if i_t == os:
                        new_os = len(sents)
                        tokens_wordpiece = [obj_type]

            elif input_format == 'entity_marker':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['[E1]'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['[/E1]']
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ['[E2]'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ['[/E2]']

            elif input_format == 'entity_marker_punct':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['@'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ['#'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ['#']

            elif input_format == 'typed_entity_marker':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = [subj_start] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + [subj_end]
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = [obj_start] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + [obj_end]

            elif input_format == 'typed_entity_marker_punct':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['@'] + ['*'] + subj_type + ['*'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ["#"] + ['^'] + obj_type + ['^'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ["#"]

            sents.extend(tokens_wordpiece)
        sents = sents[:self.args.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        return input_ids, new_ss + 1, new_os + 1


class TACREDProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}

    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = json.load(fh)

        for d in tqdm(data):
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens = d['token']
            tokens = [convert_token(token) for token in tokens]

            input_ids, new_ss, new_os = self.tokenize(tokens, d['subj_type'], d['obj_type'], ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['relation']]

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
            }

            features.append(feature)
        return features


class SemevalProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.LABEL_TO_ID = {"Other":0,"Cause-Effect":1, "Component-Whole":2, "Entity-Destination":3, "Entity-Origin":4, "Product-Producer": 5, "Member-Collection":6, "Message-Topic": 7, "Content-Container":8, "Instrument-Agency":9}
        self.ID_TO_LABEL = {i:k for k,i in self.LABEL_TO_ID.items()}
    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = [json.loads(line.strip()) for line in fh]

        for d in tqdm(data):
            ss, se = d['ner'][0][0][0], d['ner'][0][0][1]
            os, oe = d['ner'][0][1][0], d['ner'][0][1][1]

            tokens = d['sentences'][0]
            tokens = [convert_token(token) for token in tokens]

            input_ids, new_ss, new_os = self.tokenize(tokens, 'E', 'E', ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['relations'][0][0][4]]

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
            }

            features.append(feature)
        return features

class SciercProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.LABEL_TO_ID = {"NONE": 0, "PART-OF": 1, "USED-FOR": 2, "FEATURE-OF": 3, "CONJUNCTION": 4, "EVALUATE-FOR": 5, "HYPONYM-OF": 6, "COMPARE": 7}
        self.ID_TO_LABEL = {i:k for k,i in self.LABEL_TO_ID.items()}
    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = [json.loads(line.strip()) for line in fh]

        for d in tqdm(data):
            ss, se = d['ner'][0][0][0], d['ner'][0][0][1]
            os, oe = d['ner'][0][1][0], d['ner'][0][1][1]

            tokens = d['sentences'][0]
            tokens = [convert_token(token) for token in tokens]

            input_ids, new_ss, new_os = self.tokenize(tokens, d['ner'][0][0][2], d['ner'][0][1][2], ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['relations'][0][0][4]] if len(d['relations'][0]) > 0 else self.LABEL_TO_ID['NONE']
            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
            }

            features.append(feature)
        return features

class AceProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.LABEL_TO_ID = {"NONE":0,"PHYS":1,"GEN-AFF":2,"PER-SOC":3,"ORG-AFF":4,"PART-WHOLE":5,"ART":6}
        self.ID_TO_LABEL = {i:k for k,i in self.LABEL_TO_ID.items()}
    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = [json.loads(line.strip()) for line in fh]

        for d in tqdm(data):
            ss, se = d['ner'][0][0][0], d['ner'][0][0][1]
            os, oe = d['ner'][0][1][0], d['ner'][0][1][1]

            tokens = d['sentences'][0]
            tokens = [convert_token(token) for token in tokens]

            input_ids, new_ss, new_os = self.tokenize(tokens, d['ner'][0][0][2], d['ner'][0][1][2], ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['relations'][0][0][4]] if len(d['relations'][0]) > 0 else self.LABEL_TO_ID['NONE']
            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
            }

            features.append(feature)
        return features


class TBDProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.LABEL_TO_ID = {"AFTER":0, "SIMULTANEOUS":1, "VAGUE":2, "INCLUDES":3, "BEFORE":4, "IS_INCLUDED":5}
        self.ID_TO_LABEL = {i:k for k,i in self.LABEL_TO_ID.items()}
    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = [json.loads(line.strip()) for line in fh]

        for d in tqdm(data):
            ss, se = d['ner'][0][0][0], d['ner'][0][0][1]
            os, oe = d['ner'][0][1][0], d['ner'][0][1][1]

            tokens = d['sentences'][0]
            tokens = [convert_token(token) for token in tokens]

            input_ids, new_ss, new_os = self.tokenize(tokens, 'E', 'E', ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['relations'][0][0][4]] if len(d['relations'][0]) > 0 else self.LABEL_TO_ID['VAGUE']
            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
            }

            features.append(feature)
        return features