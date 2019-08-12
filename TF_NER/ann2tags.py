class Ann2Tag(object):
    def __init__(self,
                 tokenizer):
        self._tokenizer = tokenizer

    @staticmethod
    def convert_helper(text, ents):
        last_idx = 0
        obj = []

        for ent in ents:
            span = ent['span']

            obj.append({'text':text[last_idx:span[0]], 'type':'O'})
            obj.append({'text': text[span[0]:span[1]], 'type': ent['type']})

            last_idx = span[1]

        obj.append({'text':text[last_idx:-1], 'type':'O'})

        return list(filter(lambda x: x['text'] != '', obj))

    def convert(self, obj):
        text = obj['text']
        tokens = []
        tags = []
        converted_obj = Ann2Tag.convert_helper(text, obj['entities'])

        for item in converted_obj:
            t_ = item['text']
            type_ = item['type']
            temp_tokens = self._tokenizer.tokenize(t_)

            if type_ == 'O':
                tags += ['O']*len(temp_tokens)
            else:
                if len(temp_tokens) >= 2:
                    tags += ['B-{}'.format(type_)]
                    tags += ['I-{}'.format(type_)] * (len(temp_tokens) - 2)
                    tags += ['E-{}'.format(type_)]
                elif len(temp_tokens) == 1:
                    tags += ['S-{}'.format(type_)]

            tokens += temp_tokens

        return {'tokens':tokens, 'tags':tags}


